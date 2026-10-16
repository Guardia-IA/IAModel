# C1 — Pipeline de pose (producción piloto) vs entrenamiento IAModel

Documento autocontenido para diseñar un **núcleo compartido** producción + offline.  
**Repo:** `kanvis-monitoring` · **Config principal:** `config/monitoring.yaml` · **Docker:** `.env.docker` (plantilla `.env.docker.example`).

**Fecha de referencia:** valores del piloto en `monitoring.yaml` del repo; en despliegue manda **`.env.docker` sobre YAML** donde exista override documentado.

---

## Referencia entrenamiento (IAModel)

| Paso | Descripción |
|------|-------------|
| Vídeo | Recorte/reescala **FFmpeg** |
| Inferencia | **Ultralytics** `.track(half=True, conf por defecto)` |
| Tracker | **BoT-SORT + ReID** (`new_track_thresh` 0.7, `track_buffer` 300, `match_thresh` 0.5, GMC `sparseOptFlow`) |
| Frames | Sin ID → **descartados** |
| Keypoints | **8** COCO (5–12), **xyn**, conf ≤ **0.5** → **NaN** |
| `valid_mask` | ≥ **5** KP visibles |
| FPS | **Todos** los frames del vídeo (sin submuestreo) |

---

## Divergencia crítica (piloto actual)

| Aspecto | Entrenamiento | C1 producción (piloto) |
|---------|---------------|-------------------------|
| Inferencia | Ultralytics local | **Triton** `tensorrt_plan` (`yolo26s-pose`) |
| Tracker | BoT-SORT + ReID | **ByteTrack** (`bytetrack_long.yaml`) |
| FPS infer | Todos | **~15** sesión (`pose_fps`), **~4** idle; no todos los frames RTSP |
| Timeline ingest | — | Posible **hold-last** (`is_synthetic`) entre inferencias |
| Export MinIO | 0.5 / 5 KP | **IAModel** solo si `CONTINUOUS_POSE_EXPORT_IAMODEL_FULL=true` |

`src/kanvis_monitoring/pose/guadia_custom_tracker.yaml` está alineado con IAModel pero **no está activo** (`tracking.tracker_config_file: null`).

---

## Mapa de código (dónde vive cada fase)

| Fase | Módulos / funciones |
|------|---------------------|
| Ingest RTSP | `stream_defaults` + drivers RTSP/GStreamer |
| Colas / FPS / live-edge | `workers/continuous/pipeline.py` |
| Resize pre-letterbox | `pose/detector.py` (`_prepare_infer_image`) |
| Letterbox + Triton | `pose/triton/preprocess.py`, `pose/triton/client.py` |
| Post-decode TRT | `pose/triton/postprocess.py` (`decode_triton_pose_output`, `_decode_end2end_pose_output`) |
| Tracker | `pose/detector.py` (`YOLO.track`), yaml `pose/bytetrack_long.yaml` |
| Sesiones / tracks | `workers/continuous/track_sessions.py` |
| ROI (lógica, no crop imagen) | `workers/continuous/roi_context.py`, `pipeline.py` |
| Export NPY / MinIO | `pose/guadia_export.py`, `transport/minio_chunk_store.py` |
| Pool Triton multi-cámara | `pose/triton_pool.py` |
| Triton `config.pbtxt` | `scripts/triton_pose_config.py` |
| Plan TRT | `scripts/triton_pose_entrypoint.sh` (`trtexec`) |
| Offline parcial | `scripts/simulate_continuous_session.py`, `scripts/offline_video_to_chunks.py` |

---

## 1. Entrada de imagen

### Comportamiento

- **URL RTSP** por cámara desde backend (`CameraSourceConfig.url`). Substream vs main **no lo fija el repo**; lo define el catálogo/NVR.
- Tras decode, **resize** a `pose.infer_frame_max_width` × `pose.infer_frame_max_height` (**640×360** en piloto) antes de letterbox/Triton.
- **No hay recorte de imagen** para YOLO. ROI/región filtra **tracks y sesiones**, no el tensor de entrada.
- Decode preferido: **GStreamer + NVDEC** (`stream_defaults.enable_nvdec: true`).
- Espacio de color hacia OpenCV: **BGR `uint8`**; hacia Triton: **NCHW**, `/255`, FP16 en gRPC si está activo.

### Configuración (`monitoring.yaml`)

**`frame_sampling`** (fase trigger legacy; resolución referencia):

```yaml
trigger_frame_width: 640
trigger_frame_height: 360
```

**`pose`:**

```yaml
infer_frame_max_width: 640
infer_frame_max_height: 360
tracking_imgsz: 640
keypoint_indices_enabled: [5, 6, 7, 8, 9, 10, 11, 12]
```

**`stream_defaults`:**

```yaml
preferred_ingest_backend: gstreamer
enable_nvdec: true
gstreamer_force_software_h264_decode: false
rtsp_latency_ms: 250
ffmpeg_rtsp_transport: tcp
```

**`interaction`** (vídeo sesión / cadencia legacy fase 2):

```yaml
capture_fps: 12.0
capture_record_full_stream: true
capture_enforce_fps_in_full_stream: false
max_frame_width: 1920
max_frame_height: 1080
```

En **continuo v2**, la inferencia pose sigue `continuous.pose_fps` (ver §5), no `interaction.capture_fps`.

---

## 2. Preproceso

| Parámetro | Valor piloto | Código |
|-----------|--------------|--------|
| Letterbox `imgsz` | **640×640** | `pose.tracking_imgsz` |
| Pad color | **114** | `letterbox_cpu` / `letterbox_gpu` |
| Interpolación resize | **LINEAR** | OpenCV / `F.interpolate` |
| Normalización | `/255.0`, **NCHW** | `preprocess.py` |
| Entrada Triton gRPC | **FP16** si `triton_input_fp16: true` | `pose/triton/client.py` |
| Dónde se ejecuta | **Cliente C1** (CPU OpenCV por defecto) | `triton_server_letterbox_enabled: false` |
| Workers preprocess | **8** (+ dedicated 12 en lane dedicada) | `triton_preprocess_workers` |

**`pose` (extracto Triton / preprocess):**

```yaml
inference_backend: triton
triton_url: triton-pose:8011
triton_model_name: yolo26s-pose
triton_platform: tensorrt_plan
triton_server_letterbox_enabled: false
triton_max_batch_size: 32
triton_max_queue_delay_microseconds: 3000
triton_input_fp16: true
triton_system_shm_enabled: true
triton_preprocess_workers: 8
triton_gpu_preprocess_enabled: false
triton_onnx_end2end: true
```

---

## 3. Modelo y post-proceso

### Triton (piloto)

- Modelo: **`yolo26s-pose`**, plataforma **`tensorrt_plan`**.
- ONNX export end2end → salida **`(batch, 300, 57)`** (top-300, NMS en engine): caja + conf + clase + 17×3 KP.
- Entrada: nombre típico **`images`**, dims **`[3, 640, 640]`**, FORMAT_NCHW.

### Umbrales C1

| Uso | Conf detección | IoU NMS (solo modo no-e2end) |
|-----|----------------|------------------------------|
| Default pose | `pose.confidence_threshold` **0.30** | `pose.iou_threshold` **0.45** |
| Idle continuo | `continuous.pose_confidence_threshold_idle` **0.22** | — |
| Sesión activa | `continuous.pose_confidence_threshold_active` **0.18** | — |
| Tracking Ultralytics | `tracking.confidence_threshold` **0.22** | `tracking.iou_threshold` **0.45** |

Post-proceso end2end: filtra `score >= conf_threshold`; **no** filtra clase explícitamente (modelo pose = persona).  
Reescala KP/cajas con `LetterboxMeta.inverse_*` → coords en frame **640×360** (post `_prepare_infer_image`).

### Export a NPY (normalización)

- Subset **`KEEP_KPS`** = COCO **5–12** (8 joints): `pose/guadia_export.py`.
- Normalización **x/w, y/h** en **[0, 1]** sobre dimensiones de frame de inferencia (`_packet_to_keep_kps_row_like_xyn`).
- **Perfil IAModel (MinIO):** conf KP ≤ **0.5** → NaN; `valid_mask` True si ≥ **5** KP visibles (sin NaN), excluye frames `is_synthetic`.

**Constantes en código (`guadia_export.py`):** `MIN_KP_CONF` / `BODY_VISIBLE_MIN_KPS` usados según perfil; en IAModel full mandan `ContinuousConfig.minio_pose_export_thresholds()` → **0.5** y **5**.

---

## 4. Tracker

### Producción (activo)

**`monitoring.yaml`:**

```yaml
tracking:
  algorithm: bytetrack
  tracker_config_file: null
  use_tracker_in_continuous: true
  confidence_threshold: 0.22
  iou_threshold: 0.45
```

**`src/kanvis_monitoring/pose/bytetrack_long.yaml`:**

```yaml
tracker_type: bytetrack
track_high_thresh: 0.3
track_low_thresh: 0.05
new_track_thresh: 0.25
track_buffer: 9000
match_thresh: 0.92
fuse_score: true
```

- **`YOLO.track()`** en continuo (`PoseDetector`, `use_tracker_in_continuous: true`).
- **Un estado de tracker por cámara**; inferencia Triton puede batchear **varias cámaras** (`triton_pool.py`, InferBatchCollector F7).
- **No** se replica la regla offline “descartar frame sin track ID”; la sesión se gestiona en `track_sessions.py`.

### IAModel (referencia, no activo en piloto)

**`src/kanvis_monitoring/pose/guadia_custom_tracker.yaml`:**

```yaml
tracker_type: botsort
new_track_thresh: 0.7
track_buffer: 300
match_thresh: 0.5
gmc_method: sparseOptFlow
with_reid: True
model: models/yolo26n-cls.pt
```

**Identidad ReID global:** `identity.reid_enabled: false` (piloto tubo julio/ByteTrack sin OSNet en continuo).

---

## 5. Tiempo y frames

### Cadencia

**`continuous` (piloto):**

```yaml
pose_fps: 15.0
pose_fps_match_stream: true
pose_fps_max: 0.0
pose_idle_scan_fps: 4.0
pose_timeline_decouple_infer: false
pose_infer_every_video_frame_during_session: true
pose_session_drain_max_frames_per_cycle: 4
```

- **Inferencia pose en sesión:** objetivo **~15 fps** al ritmo del stream (no 12).
- **12 fps** en `interaction.capture_fps` aplica sobre todo a contrato **legacy** / cabecera MP4; en v2 el clip sale del **rolling buffer** (`session_video_from_rolling_buffer: true`, `session_video_gap_policy: pack`).

### Timestamps en `.npy`

- `pose_frame_timestamps.npy`: **epoch UTC** (`packet.ts`) por fila de chunk — `chunk_pose_frame_timestamps_seconds()` en `guadia_export.py`.
- No es PTS RTSP crudo en el artefacto MinIO.

### Colas y descartes

```yaml
pose_queue_max_lag_frames: 900
pose_idle_queue_max_frames: 24
pose_queue_live_edge_drop: false
pose_queue_backpressure_enabled: true
pose_session_backlog_mode: true
pose_session_backlog_max_seconds: 60.0
```

- **Idle:** live-edge (drop-oldest) con tope **24** frames.
- **Sesión:** backpressure / backlog; no live-edge agresivo por defecto.

### Hold-last vs NaN

- Frames sin inferencia YOLO pueden marcarse **`is_synthetic`** (hold-last mientras hay persona): `pipeline.py` (`_ingest_skipped_video_frames_hold_last`).
- **`continuous.pose_gap_policy: nan`** — en export medido, hold-last no cuenta como presencia real en `valid_mask` (IAModel).

### Gap-fill en export MinIO

| Perfil | `pose_chunk_gap_fill` efectivo |
|--------|--------------------------------|
| YAML default (legacy) | **`interpolate`** (+ `pose_sparse_interpolate_max_ratio: 0.35`) |
| IAModel full | **`none`** (`pose_export_iamodel_gap_fill: ""` → `minio_pose_chunk_gap_fill()`) |

**Precedencia IAModel:** env `.env.docker` > YAML (ver abajo).

---

## 6. Export (chunks y sesiones)

### Ventanas

```yaml
window_seconds: 8.0
stride_seconds: 4.0
c2_inference_stride_seconds: 4.0
publish_all_pose_chunks: true
```

### Umbrales export vs gates de sesión

| Ámbito | min KP conf | min KP visibles | Fuente |
|--------|-------------|-----------------|--------|
| Gates sesión / apertura | 0.35 | 2 | `continuous.pose_export_min_kp_conf`, `pose_export_min_visible_kps` |
| Sesión activa (relajado) | 0.25 | — | `pose_export_min_kp_conf_active`, etc. |
| **MinIO IAModel** | **0.5** | **5** | `pose_export_iamodel_*` + env `CONTINUOUS_POSE_EXPORT_*` |

### Reglas de filas

- **`poses_full.npy`:** una fila por frame del chunk; **NaN** si no hay medición o conf baja (IAModel).
- **`valid_mask.npy`:** bool por fila; **False** si &lt; 5 KP o frame sintético (IAModel).
- **`poses.npy`:** subconjunto `poses_full[valid_mask]` (pre gap-fill medido).
- **Track omitido en un frame:** **no se omite la fila temporal**; va como NaN en `poses_full`.

### Sesión / personas

```yaml
session_open_max_persons_in_frame: 0
session_subject_absent_seconds: 3.0
session_exportable_gap_close_seconds: 15.0
session_max_wall_seconds: 120.0
```

Lógica detallada: `workers/continuous/track_sessions.py`.

### Artefactos MinIO por chunk

- `poses_full.npy`, `valid_mask.npy`, `keypoint_mask.npy`, `pose_frame_timestamps.npy`, `meta.json`
- Con versionado: `pose_bundle.json`, campos `pose_bundle_id`, `iamodel_full`, umbrales en `meta.json`
- Doc export IAModel: `docs/POSE_EXPORT_IAMODEL.md`

---

## 7. Rendimiento y paralelismo Triton

### `config.pbtxt` (generado)

Plantilla en `scripts/triton_pose_config.py` — valores piloto desde YAML:

| Campo | Valor típico piloto |
|-------|---------------------|
| `max_batch_size` | **32** |
| `dynamic_batching.max_queue_delay_microseconds` | **3000** |
| `preferred_batch_size` | 4, 8, 16, 32 |
| `preserve_ordering` | **true** |
| `instance_group.count` (GPU) | **2** |
| Input `data_type` | **TYPE_FP16** si `triton_input_fp16: true` |
| Output | **TYPE_FP32** |

### Batch cross-cámara

- **`pose.triton_infer_batch_collector_enabled`** (F7): agrega frames de varias cámaras antes de gRPC (`triton_pool.py`).
- Postprocess y tracker mantienen **contexto por cámara** (order gate / dedicated lane).

### `trtexec` (entrypoint sidecar)

`scripts/triton_pose_entrypoint.sh`:

- `--onnx=model.onnx`, `--saveEngine=model.plan`
- `--minShapes=images:1x3x640x640`
- `--optShapes=images:16x3x640x640` (opt batch min(16, max_batch))
- `--maxShapes=images:32x3x640x640`
- `--fp16` si `build_spec.fp16`
- `--inputIOFormats=fp16:chw` si `input_fp16`
- `--timingCacheFile=trt_timing.cache`

Imagen sidecar por defecto: `nvcr.io/nvidia/tritonserver:25.02-py3` (ver compose).

### Arranque Docker (sin mount engine host)

```bash
docker compose -f docker-compose.pilot-server.yml -f docker-compose.pilot-server.gst.yml \
  --env-file .env.docker up -d --build --no-deps triton-pose kanvis-monitoring minio
```

C1: export ONNX + sync repo → `triton-pose` compila `model.plan` si cambió ONNX/`build_spec`.

---

## 8. Variables de entorno (`.env.docker`)

Plantilla: **`.env.docker.example`**. Las que afectan **directamente** al contrato pose/export:

| Variable | Efecto |
|----------|--------|
| `CONTINUOUS_POSE_EXPORT_IAMODEL_FULL` | `true` → umbrales MinIO 0.5 / 5 KP, gap-fill `none` |
| `CONTINUOUS_POSE_EXPORT_MIN_KP_CONF` | Override conf export IAModel |
| `CONTINUOUS_POSE_EXPORT_MIN_VISIBLE_KPS` | Override mín. KP visibles export |
| `CONTINUOUS_POSE_EXPORT_IAMODEL_GAP_FILL` | p. ej. `none` |
| `KANVIS_POSE_BUNDLE_ID` | `pose_bundle_id` en meta/chunk (default yaml: `yolo26s-pose_v1`) |
| `KANVIS_POSE_BUNDLE_PLAN_PATH` | SHA plan en manifiesto (opcional) |
| `KANVIS_POSE_BUNDLE_PT_PATH` | SHA `.pt` en manifiesto (opcional) |
| `KANVIS_POSE_INFERENCE_BACKEND` | Override `pose.inference_backend` |
| `KANVIS_POSE_TRITON_URL` | Override URL gRPC Triton |
| `KANVIS_POSE_TRITON_MODEL` | Override nombre modelo |
| `KANVIS_POSE_TRITON_ONNX_END2END` | Override export end2end |
| `KANVIS_POSE_TRITON_INPUT_FP16` | Override FP16 entrada |
| `KANVIS_OBJECT_STORE_*` | MinIO (chunks) |
| `TRITON_POSE_IMAGE` | Imagen sidecar (compose) |

**Nota:** Tras revertir mount de `.engine` externo, **no** hay variables `YOLO_POSE_ENGINE*` en el flujo estándar.

### Bloque recomendado piloto (export IAModel)

```bash
CONTINUOUS_POSE_EXPORT_IAMODEL_FULL=true
CONTINUOUS_POSE_EXPORT_MIN_KP_CONF=0.5
CONTINUOUS_POSE_EXPORT_MIN_VISIBLE_KPS=5
# CONTINUOUS_POSE_EXPORT_IAMODEL_GAP_FILL=none
KANVIS_POSE_BUNDLE_ID=yolo26s-pose_v1
```

---

## 9. Tabla resumen YAML `continuous` + `pose` (piloto)

### `pose` (inferencia)

| Clave | Valor |
|-------|-------|
| `model_name` | `models/yolo26s-pose.engine` (referencia; infer vía Triton) |
| `confidence_threshold` | 0.30 |
| `infer_frame_max_width/height` | 640 / 360 |
| `tracking_imgsz` | 640 |
| `inference_backend` | triton |
| `triton_model_name` | yolo26s-pose |
| `triton_onnx_end2end` | true |
| `triton_max_batch_size` | 32 |
| `triton_gpu_instance_count` | 2 |
| `track_half_precision` | true (local .engine; Triton usa plan TRT) |

### `continuous` (export + tiempo)

| Clave | Valor |
|-------|-------|
| `enabled` | true |
| `window_seconds` / `stride_seconds` | 8 / 4 |
| `pose_fps` | 15 |
| `pose_export_min_kp_conf` / `pose_export_min_visible_kps` | 0.35 / 2 |
| `pose_export_iamodel_full` | **false** en YAML (usar env **true** en piloto IAModel) |
| `pose_export_iamodel_min_kp_conf` / `min_visible_kps` | 0.5 / 5 |
| `pose_chunk_gap_fill` | interpolate |
| `pose_bundle_id` | yolo26s-pose_v1 |
| `pose_gap_policy` | nan |

---

## 10. Reutilización offline = prod

| Pieza | ¿Reutilizable sin MinIO/colas live? |
|-------|--------------------------------------|
| `guadia_export.py` | **Sí** — contrato `.npy` |
| `preprocess.py` + `postprocess.py` + Triton client | **Sí** |
| `pipeline.py` + `triton_pool.py` | **No** tal cual — acoplados a RTSP, colas, backlog, multi-cámara |
| `simulate_continuous_session.py` | Parcial — pipeline continuo simplificado |
| `offline_video_to_chunks.py` | Parcial — **YOLO local**, no Triton |

**Dependencias de estado global:** reloj de pared, hold-last ingest, batch cross-cámara, política de colas.

---

## 11. Artefacto de prueba (pedido coordinación)

**Estado actual:** no hay dump estándar pre-track en un solo folder.

**Propuesta P0:**

1. Vídeo corto compartido entre equipos.
2. C1 con Triton READY + `CONTINUOUS_POSE_EXPORT_IAMODEL_FULL=true` + `KANVIS_POSE_BUNDLE_ID`.
3. Entregar prefijo MinIO del chunk + (futuro) `debug_detections.jsonl` / `debug_tracks.jsonl` — **pendiente de implementar**.

Scripts existentes:

- `scripts/simulate_continuous_session.py` — más cercano a continuo v2.
- `scripts/offline_video_to_chunks.py` — chunks sin Triton.
- `scripts/check_triton_pose_batch.py` — smoke 640×640 contra Triton.

---

## 12. Checklist paridad con entrenamiento (gap analysis)

- [ ] Tracker: ByteTrack → BoT-SORT+ReID (o documentar decisión de mantener ByteTrack en prod).
- [ ] FPS: 15 infer + hold-last vs todos los frames FFmpeg.
- [ ] Resolución: 640×360 + letterbox 640 vs pipeline FFmpeg entrenamiento.
- [ ] Inferencia: Triton end2end vs Ultralytics `.track()` local.
- [ ] Export: activar IAModel en prod (`CONTINUOUS_POSE_EXPORT_IAMODEL_FULL=true`).
- [ ] Núcleo compartido: extraer cadena letterbox → Triton → decode → track → `guadia_export`.
- [ ] Debug export pre-track para cruce numérico con `pose_extractor_clean.py`.

---

## Referencias cruzadas

- `docs/POSE_EXPORT_IAMODEL.md` — perfil MinIO IAModel y runbook Triton ONNX→plan.
- `docs/KANVIS_V2_CONTINUO.md` — visión continuo v2 (si aplica en repo).
- `config/monitoring.yaml` — fuente de verdad estática.
- `.env.docker.example` — overrides Docker piloto.
