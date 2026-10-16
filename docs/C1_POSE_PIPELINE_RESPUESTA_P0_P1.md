# C1 — Respuesta P0/P1 (pipeline pose compartido)

Complemento de [`C1_POSE_PIPELINE_HANDOFF.md`](C1_POSE_PIPELINE_HANDOFF.md) para cerrar diseño del núcleo compartido y plan de validación.  
**Repo:** `kanvis-monitoring` · **Config:** `config/monitoring.yaml` + overrides `.env.docker`.

---

## P0 — Bloquean el diseño

### 1. Orden de canales hacia Triton (BGR vs RGB)

**Conclusión:** en el camino de producción **no hay conversión BGR→RGB** antes del tensor Triton. OpenCV entrega **BGR `uint8`**; el preprocess hace `HWC → CHW` y `/255` **sin permutar canales**.

Fragmento relevante (`src/kanvis_monitoring/pose/triton/preprocess.py`):

```36:47:src/kanvis_monitoring/pose/triton/preprocess.py
def chw_normalized(boxed: np.ndarray, *, input_fp16: bool) -> np.ndarray:
    """HWC uint8 -> CHW normalizado y contiguo.
    ...
    """
    if input_fp16:
        return _U8_TO_UNIT_FP16[np.ascontiguousarray(boxed.transpose(2, 0, 1))]
    return np.ascontiguousarray(boxed.transpose(2, 0, 1).astype(np.float32) / 255.0)
```

Letterbox también opera en BGR (`letterbox_cpu` → `cv2.resize` sobre imagen BGR).

**Implicación frente a Ultralytics local:** el entrenamiento/inferencia Ultralytics suele convertir a **RGB** antes del backbone. Si el ONNX/TRT se exportó desde Ultralytics con el mismo preproceso estándar, alimentar **orden BGR** en Triton introduce un **sesgo sistemático de canales** (R↔B). Acciones posibles para paridad: `boxed[..., ::-1]` antes de `transpose`, o re-export ONNX verificando el grafo de color, o benchmark A/B offline.

---

### 2. Cómo se alimenta el tracker (Triton vs YOLO local)

| Pregunta | Respuesta |
|----------|-----------|
| ¿Inferencia local con `.engine` en prod? | **No** cuando `pose.inference_backend: triton`. No se carga el `.engine`; inferencia = gRPC Triton. |
| ¿Para qué sirve `pose.model_name` / `.pt`? | Solo **ByteTrack**: se carga `YOLO(.pt)` en **CPU** para instanciar predictor + tracker Ultralytics, **sin** forward del detector local. |
| ¿Qué recibe el tracker? | Un `MockYoloResult` con **cajas + conf + 17 keypoints** (coords ya en espacio del frame letterbox/inferencia). ByteTrack usa sobre todo cajas; la API es `tracker.update(result, img=prepared_image)` como en Ultralytics. |
| Versión Ultralytics | **`ultralytics==8.4.42`** (`pyproject.toml`, extra `yolo`). |

Cadena concreta (Triton → tracker → personas):

```640:717:src/kanvis_monitoring/pose/detector.py
    def build_results_from_triton_batch(
        ...
    ) -> list[PoseFrameResult]:
        """Post-proceso Triton: decode + ByteTrack secuencial (orden del batch)."""
        ...
            decoded = decode_triton_pose_output(
                raw_out,
                letterbox=meta,
                conf_threshold=max(0.05, parse_conf * 0.5),
                iou_threshold=float(self.config.iou_threshold),
            )
            ...
                    tracked = self._apply_tracker_to_predict_result(
                        decoded,
                        prepared_image=prep,
                        camera_id=camera_id,
                        profile=profile,
                    )
            ...
                results.append(
                    self._pose_frame_result_from_raw(frame, tracked, profile=profile)
                )
```

Carga tracker-only:

```610:631:src/kanvis_monitoring/pose/detector.py
    def _ensure_tracker_model_loaded(self) -> None:
        """YOLO .pt en CPU solo para ByteTrack cuando la inferencia va por Triton."""
        ...
                self._model = YOLO(pt_path)
                logger.info(
                    "triton_tracker_model_loaded path=%s note=cpu_tracker_only",
                    pt_path,
                )
```

`MockYoloResult` (`postprocess.py`) incluye `boxes` + `keypoints` (17×2 + conf) antes de `_apply_tracker_to_predict_result`.

**Nota:** `YOLO.track()` en docstrings del handoff describe el **modo local** (`_infer_batch_temporal_tracked`); en piloto Triton el tracker es el mismo código pero alimentado por decode Triton, no por `predict()` del `.engine`.

---

### 3. Umbrales que llegan al tracker (conf 0.30 / 0.22 / 0.18)

Hay **tres capas** distintas:

| Capa | Umbral efectivo (piloto) | Dónde |
|------|---------------------------|--------|
| **A. Decode Triton (NMS end2end)** | `max(0.05, infer_conf × 0.5)` | `build_results_from_triton_batch` |
| **B. Filtro al parsear personas** | `infer_conf` del frame | `_parse_one_result` (`conf_val < parse_conf` → descarte) |
| **C. Bootstrap `predict()` local** | `tracking.confidence_threshold` **0.22** | Solo si se usa inferencia local / arranque predictor; **no** sustituye A–B en Triton |

**`infer_conf` en continuo** (inyectado en cada frame antes del pool Triton):

```2003:2012:src/kanvis_monitoring/workers/continuous/pipeline.py
    def _continuous_infer_conf(w: "CameraWorker", sessions: TrackSessionManager) -> float:
        """Conf estricta en sesión activa; permisiva en idle/scan pre-orchestrator."""
        ...
        if active_conf > 0 and ContinuousPipeline._orchestrator_sessions_active(sessions) > 0:
            return active_conf   # pose_confidence_threshold_active → 0.18
        ...
            return idle_conf     # pose_confidence_threshold_idle → 0.22
        return float(w.pose_config.confidence_threshold)  # 0.30
```

**En sesión activa con Triton (valores piloto):**

- Decode end2end: **≥ 0.09** (`max(0.05, 0.18×0.5)`).
- Personas exportadas al pipeline: cajas **≥ 0.18** (`parse_conf`).

**ByteTrack** (`bytetrack_long.yaml`): `track_low_thresh: 0.05`, `new_track_thresh: 0.25`, `track_high_thresh: 0.3`.

**¿Queda anulada la asociación low-conf de ByteTrack?** Parcialmente: las detecciones que entran al tracker ya pasaron el decode a **~0.09+** y luego `_parse_one_result` a **0.18** en sesión activa. Detecciones entre **0.05–0.18** **no llegan** al tracker en activa, así que la rama “segunda pasada” low de ByteTrack **casi no tiene material** salvo que baje `infer_conf` o el factor `×0.5` del decode. En **idle** (`infer_conf=0.22`): decode **≥0.11**, parse **≥0.22**.

`pose.confidence_threshold` **0.30** aplica cuando no hay overrides idle/active (p. ej. trigger legacy o sin `infer_conf` en frame).

---

### 4. Frames con track no confirmado → `poses_full`

**No replicamos** la regla offline “descartar frame sin ID”.

- Tras tracker, si `boxes.id` es `None`, `_parse_one_result` asigna **`track_id` estable por orden izquierda→derecha** (`"0"`, `"1"`, …), no descarta el frame.
- **Toda** detección que pasa ingest genera **una fila** en la ventana deslizante (`sliding_window.append`), aunque los keypoints no sean “exportables”.
- Filas sin keypoints válidos van con `keypoints_keep_kps={}` → en MinIO **`poses_full` = NaN** en IAModel; **`valid_mask` = False** si aplica sintético o pocos KP.

Antes de `new_track_thresh` (0.25): si ByteTrack aún no asigna ID entero, puede usarse el fallback por rank; en cuanto hay ID, la sesión sigue el mismo `track_id` en metadatos del chunk.

**Inicio de chunk:** no está acoplado al instante de “confirmación” ByteTrack, sino a **apertura/publicación de sesión** + ventana `window_seconds` / `stride_seconds`. Los primeros frames de una sesión pueden ser filas NaN hasta haber pose exportable.

---

### 5. Semántica de sesiones, export y ROI

#### Claves (valores piloto `monitoring.yaml`)

| Clave | Valor | Semántica |
|-------|-------|-----------|
| `session_open_max_persons_in_frame` | **0** | **0 = sin límite.** Solo bloquea si `limit > 0` y `persons_in_frame > limit` (`_session_open_blocked_by_person_count`). |
| `session_subject_absent_seconds` | **3** | Cierra si el **sujeto de la sesión** deja de verse (track, bbox IoU, ReID) durante **3 s** — aunque el esqueleto exportable falle. |
| `session_exportable_gap_close_seconds` | **15** | En sesiones **ya publicadas** (`orchestrator_started`): cierra si llevan **>15 s** sin pose exportable **y** el sujeto no está visible (modo estricto). |
| `session_max_wall_seconds` | **120** | Tope duro de duración de sesión desde `started_at` (120 s); con ≥2 personas en escena puede aplicarse cap multi-persona si está configurado). |

#### Ejemplo narrativo (una visita)

1. **T=0:** una persona entra; ByteTrack id **7**; se abre `TrackSession` con `session_id` UUID (orquestador).
2. **T=0–5 s:** inferencia 15 fps; filas en chunk con mezcla de NaN y poses válidas; `exportable_pose_frames` sube cuando ≥ umbrales de apertura (0.35/2 KP) o activos (0.25/…).
3. **T=40 s:** id **7** se pierde, id **12** continúa la misma persona (ReID/bbox): `align_session_track_for_ingest(..., force=True)` puede **rebind** `track_id` de la sesión de 7→12 **sin** nueva sesión si pasa `_rebind_ready`.
4. **T=45 s:** sin bbox del sujeto **3 s** → cierre por `session_subject_absent_seconds`.
5. **T=45 s + wall 120** no alcanzado → sesión termina por ausencia, no por wall.

#### ¿Un `.npy` por `track_id` o por sesión?

- **Una sesión de persona** (`session_id`) → varios **chunks** (`chunk_000`, `chunk_001`, …) bajo prefijo MinIO `{tenant}/{store}/{camera}/{session_id}/`.
- **`track_id`** va en metadatos del chunk y puede cambiar por rebind; **no** es el primer nivel del path (`_chunk_prefix` usa `session_id`).
- Cada chunk lleva sus propios `poses_full.npy`, `valid_mask.npy`, etc.

#### Cambio de ID

- **Misma sesión** si rebind IoU/ReID lo permite (`rebind_track_id`).
- **Nueva sesión** si el orquestador abre otra interacción o cierre + nueva apertura (p. ej. swap no resuelto, multi-persona, wall).

#### Otras personas en el frame

- Pueden abrir **sesiones paralelas** (tope `session_open_max_live_slack` + personas visibles).
- Ingest va **por track**; cada sesión ingiere solo su sujeto (`_pose_ingest_block_reason`, match bbox/ReID).
- `note_unique_subjects_for_sessions` acumula identidades vistas para meta.

#### ROI (`roi_context.py` / `_tracks_inside_roi`)

- **No recorta** imagen YOLO.
- Marca tracks cuya **muñeca** (KP 9/10, conf ≥ `min_wrist_confidence_visible`) cae en polígono ROI.
- **No descarta tracks** del detector; afecta **metadatos** (`in_roi`, `region_id`) y gates downstream (actividad/publicación), no el tensor Triton.

---

### 6. Datos reales del eje temporal (20–50 chunks IAMODEL_FULL)

**Estado:** en el momento de este documento **no hay agregación automática** en repo sobre MinIO de producción. Hace falta ejecutar un script de auditoría sobre chunks con `iamodel_full: true` (campos en `meta.json` / `diag.json`: `temporal_meta.synthetic_frames`, `pose_frame_timestamps`, etc.).

**Campos ya persistidos por chunk** (`minio_chunk_store.publish_chunk` → `meta.json` + `diag.json`):

- `temporal_meta.synthetic_frames`, `sequence_gaps_total`, `mono_span_ms`, …
- Ratios en meta: `valid_mask_true_ratio`, `pose_chunk_gap_fill_applied`

**Compromiso:** entregar tabla (% `is_synthetic` media/p95, fps inferencia sesión, gaps en `pose_frame_timestamps`, duración chunks) en cuanto tengamos acceso al bucket piloto con `CONTINUOUS_POSE_EXPORT_IAMODEL_FULL=true` (script propuesto: escaneo `meta.json` + `diag.json` bajo prefijo sesión).

---

## P1 — Validación y despliegue

### 7. Determinismo batch Triton

**Estado:** **no medido** en CI ni adjunto en este envío.

**Protocolo recomendado:**

1. Fijar frame BGR uint8 (640×360 → letterbox 640).
2. Inferir batch=1 vs batches 4/16/32 con **mismo tensor repetido** y `preserve_ordering: true`.
3. Comparar `max abs diff` en keypoints xy y conf post-decode (`decode_triton_pose_output`).

Smoke existente: `scripts/check_triton_pose_batch.py` (latencia/correctitud batch, no diff numérica). Extensión: script `benchmark_triton_pose_determinism.py` (pendiente).

---

### 8. ONNX y plan — reproducibilidad

| Parámetro | Valor en repo |
|-----------|----------------|
| Ultralytics export | **`8.4.42`**, `model.export(format="onnx", imgsz=640, dynamic=True, simplify=True, opset=17, batch=max_batch, end2end=<flag>)` — `scripts/export_triton_pose_onnx.py` |
| ONNX end2end piloto | `pose.triton_onnx_end2end: true` → salida **(300, 57)** |
| `trtexec` | `--fp16`, shapes **1 / 16 / 32** × `3×640×640`, `--inputIOFormats=fp16:chw` si `build_spec.input_fp16` — `scripts/triton_pose_entrypoint.sh` |
| Triton imagen piloto | **`nvcr.io/nvidia/tritonserver:25.02-py3`** (`docker-compose.pilot-server.yml`) |
| GPU prod piloto documentada | **RTX 5090** (entrypoint log); plan **no portable** entre arquitecturas |
| SHA-256 ONNX/plan prod | Se rellenan en **`pose_bundle.json`** vía `build_pose_bundle_manifest()` (`pt_sha256`, `plan_sha256`) si los ficheros existen en sync; **no** adjuntamos hashes de wudalkot en este doc (leer del bucket o del manifiesto tras deploy) |

Variables opcionales: `KANVIS_POSE_BUNDLE_PLAN_PATH`, `KANVIS_POSE_BUNDLE_PT_PATH`, `KANVIS_POSE_BUNDLE_BUILT_AT`, `TRITON_POSE_IMAGE`.

---

### 9. Acceso offline / Triton

| Pregunta | Respuesta |
|----------|-----------|
| `triton_system_shm_enabled: true` | Requiere **cliente en mismo host** (o contenedor con `ipc: host` + `/dev/shm` compartido con Triton). Config **`pose.triton_system_shm_enabled`** en YAML; poner **`false`** fuerza gRPC sin SHM (válido para cliente remoto, más copia CPU). No hay env documentado en `.env.docker.example`; override vía YAML o extensión env en despliegue. |
| Segundo cliente gRPC | **Sí**, Triton acepta múltiples clientes gRPC en **8011**; no requiere tocar el cliente de producción si hay VRAM/cola suficientes. |
| GPU/memoria noche con `instance_group.count: 2` | Piloto YAML: **`triton_gpu_instance_count: 2`** → dos contextos TRT en la misma GPU. **Memoria libre no medida** aquí; revisar `nvidia-smi` en ventana idle + métricas Triton. |
| Segunda instancia otro `config.pbtxt` | **Factible**: otro contenedor/puerto, otro model repo, otro `model.plan` (p. ej. ONNX distinto). |

---

### 10. Empaquetado del núcleo

| Módulo | Dependencias principales |
|--------|---------------------------|
| `pose/triton/preprocess.py` | `numpy`, `opencv` (CPU), `torch` opcional (GPU preprocess) |
| `pose/triton/postprocess.py` | `numpy` |
| `pose/triton/client.py` | `numpy`, `tritonclient[grpc]`, SHM helpers |
| `pose/guadia_export.py` | `numpy`, contratos `InteractionChunk` |
| `pose/detector.py` (tracker) | **`ultralytics`**, `numpy`, `opencv` |

- **Python:** `>=3.11` (`pyproject.toml`).
- **Paquete instalable:** sí, proyecto **`kanvis-monitoring`** con setuptools; extra **`yolo`** para Ultralytics.
- **MinIO:** solo en **`transport/minio_chunk_store.py`**, no en preprocess/postprocess.
- **Estado global / reloj:** tracker state por cámara en `PoseDetector` (`_predictor_by_camera`, ByteTrack buffers); sesiones usan **`datetime` UTC** (`track_sessions`, `pipeline`). Núcleo letterbox→decode→export puede extraerse sin colas RTSP, pero el tracker **sí** es stateful por stream.

---

### 11. Resolución y aspecto

- **Nativa:** la define la URL RTSP (main/substream) en catálogo; repo fija **tope inferencia** `640×360` (`infer_frame_max_*`), no impone 16:9 en todas las cámaras.
- **`_prepare_infer_image`:** downscale con **`cv2.INTER_AREA`** si supera max width/height; **sin** letterbox (letterbox es paso aparte a 640×640).
- **Aspecto ≠ 16:9:** letterbox a 640×640 con **barras 114**; KP/cajas se reescalan a coords del frame **después** de `_prepare_infer_image` (`LetterboxMeta.orig_h/w`).

---

### 12. Ejemplos de metadatos (IAModel activo)

No adjuntamos ficheros de MinIO reales en repo. Esquemas **derivados del código** (valores ilustrativos):

**`meta.json` (chunk)** — campos clave:

```json
{
  "status": "ready",
  "schema_version": "v1",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "chunk_id": "chunk_000",
  "iamodel_full": true,
  "export_min_kp_conf": 0.5,
  "export_min_visible_kps": 5,
  "pose_bundle_id": "yolo26s-pose_v1",
  "pose_bundle_ref": "s3://bucket/tenant/store/cam/session/chunk_000/pose_bundle.json",
  "poses_full_shape": [120, 8, 2],
  "pose_frame_timestamps_unit": "utc_epoch_seconds",
  "pose_chunk_gap_fill": "none",
  "temporal_meta": {
    "synthetic_frames": 18,
    "sequence_gaps_total": 0,
    "mono_span_ms": 8000.0
  }
}
```

**`diag.json` (chunk):**

```json
{
  "schema_version": "chunk_diag_v1",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "chunk_seq": 0,
  "status": "warn",
  "issues": ["synthetic_frames_used"],
  "valid_ratio": 0.72,
  "temporal_meta": { "synthetic_frames": 18 },
  "pose_bundle_id": "yolo26s-pose_v1"
}
```

**`pose_bundle.json`:** ver `build_pose_bundle_manifest()` — incluye `plan_sha256`, `pt_sha256`, `triton_image_tag`, umbrales export, tracker config.

Tras el siguiente chunk IAModel en piloto, podemos copiar tres JSON reales anonimizados a `docs/examples/pose_chunk_iamodel/`.

---

## Artefacto de prueba — `debug_detections.jsonl` / `debug_tracks.jsonl`

**Estado:** **no implementado** en código (solo especificación acordada en handoff §11).

| Entregable | ETA orientativa |
|------------|-----------------|
| Hook en `build_results_from_triton_batch` + post-track (`PoseFrameResult`) escribiendo JSONL | **+3–5 días laborables** tras priorización |
| Clip único persona + salida `.jsonl`, `.npy`, `meta.json` | **misma ventana**, depende de Triton READY en entorno de prueba |

Esquema propuesto (aceptamos tal cual):

- **`debug_detections.jsonl`:** pre-track, coords inferencia 640×360, letterbox meta, `dets[]` con bbox/conf/kpts 17×3.
- **`debug_tracks.jsonl`:** post-track, `tracks[]` + fila export IAModel (`pose_8x2`, `kp_conf_8`, `valid`).

Hasta entonces: **`scripts/simulate_continuous_session.py`** + chunk MinIO IAModel como proxy parcial (sin JSONL pre-track).

---

## Resumen acciones para paridad / test de deriva

1. **Validar BGR vs RGB** (P0.1) con A/B numérico o fix en preprocess.
2. Documentar umbrales efectivos **0.09 / 0.18** en sesión activa Triton (P0.3).
3. Medir determinismo batch (P1.7) y publicar diffs máximas.
4. Publicar hashes desde `pose_bundle.json` post-deploy y estadísticas MinIO (P0.6).
5. Implementar dump JSONL (artefacto coordinación).

---

*Documento generado desde código en rama de trabajo del repo `kanvis-monitoring`; valores numéricos según `config/monitoring.yaml` piloto salvo indicación contraria.*
