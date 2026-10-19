# Paridad `poses_full.npy` — C1 continuo (MinIO) vs extracción offline (ClipExtractor)

**Para:** agente / equipo IAModel — extracción Triton por clip (`kanvis-triton-pose`, ClipExtractor, `PoseDataset`)  
**De:** equipo C1 — `kanvis-monitoring` (piloto wudalkot)  
**Objetivo:** que los arrays de pose de producción sean **intercambiables en contenido** con los del entrenamiento, aunque la **jerarquía en disco** sea distinta.

**Última respuesta IAModel:** 2026-10-19 (texto íntegro en [§ Anexo A](#anexo-a--respuesta-iamodel-2026-10-19)).

---

## 1. Mapeo de almacenamiento (mismo clip, distinto árbol)

| Offline (batch por clip) | C1 continuo (MinIO sesión) | Notas |
|--------------------------|----------------------------|--------|
| `clip.mp4` (~12 fps CFR en batch típico) | `clip_buffer.mp4` (+ pre/post según buffer) | Mismo rol: vídeo de referencia |
| `meta.json` | `meta.json` (sesión) + `chunk_NNN/meta.json` | C1 parte en ventanas ~8 s |
| `pose_bundle.json` | `chunk_NNN/pose_bundle.json` (y manifiesto sesión) | Mismo bundle / SHA plan |
| `user_<id>/poses_full.npy` | **`merged.npy`** (sesión) | **Equivalente semántico** (timeline del sujeto en el MP4) |
| `user_<id>/valid_mask.npy` | **`merged_valid_mask.npy`** | Debe derivar de `valid_mask` por chunk |
| `user_<id>/poses.npy` | Derivado de merged / concat chunks | `poses_full[valid_mask]` en disco offline |
| `user_<id>/keypoint_mask.npy` | Por chunk `keypoint_mask.npy` | Pendiente merged session-level |
| *(no hay chunks)* | `chunk_NNN/poses_full.npy` | Ventana parcial; **≠** unidad de entrenamiento TCN |

**Pregunta:** ¿**`user_*/poses_full` offline ≡ `merged.npy` C1**?

**Respuesta IAModel (resumen):** **Sí, con matices.** Comparar **`merged` ↔ `user_*`** sobre el **mismo `clip_buffer.mp4`**, mismo `pose_bundle_id`, mismas reglas IAModel full, **validación numérica** (T, timestamps, valid_mask). **No** equivaler **`chunk_*` ↔ `user_*`**. Entrenamiento 41|48 = **clip completo**; C2 prod en **chunks cortos** → desalineación de scoring aunque Triton sea el mismo.

---

## 2. Preguntas concretas — respuestas IAModel (2026-10-19)

### 2.1 Filas T y fps

1. Offline: **T = una fila por frame BGR decodificado** (infer **100 %** frames). No re-muestreo a 12 Hz en batch salvo que el MP4 ya sea 12 fps CFR.
2. Tras CSV+FFmpeg: suele **`frame_count == video_frame_count == len(poses_full)`** por `user_*`. Si MP4 es 15/25/30 fps, **T refleja esa fps**, no 12 fijo.
3. **`pose_frame_timestamps.npy`:** en C1/MinIO sí (epoch UTC). En batch offline clásico **a menudo no** (solo `meta.json` fps/frame_count).

**Contraste C1:** ingest ~12 Hz + hold-last + `pose_resampled` parcial → **T y fps efectivo pueden no coincidir** con offline “todos los frames del MP4”.

### 2.2 Huecos, NaN, `keep_all`

4. **`keep_all` (default):** sin detección → fila NaN + `valid_mask=False`. Con **`frame_policy: iamodel`:** track ausente → **no se añade fila** (T más corto) — no paridad grid C1 full.
5. Offline `keep_all`: **sin hold-last**; huecos = NaN (`pose_chunk_gap_fill: none`). Prod: cola coords repetidas + `valid_mask=False` (d9a71 chunk_004) **≠** offline.
6. Offline: sin `is_synthetic`. C1: excluir synthetic en `valid_mask` igual que IAModel full.

### 2.3 Loader entrenamiento (41|48)

7. **`PoseDataset` / `pose_source: full`:** lee **`poses_full.npy`** en `user_*`.
8. Antes del forward: **`poses = poses_full[valid_mask]`** (elimina filas inválidas) → normalize → add_velocity → **`temporal_resize(seq_len)`**.
9. **No** ventana fija 8 s = 96 filas. **T' = valid_mask.sum()** del clip entero, luego resize a **seq_len** del checkpoint (**~96** exp 41, **~64** exp 48). “8 s @ 12 Hz” = doc C1 continuo, **no** regla exacta del loader.

**Implicación:** train = clip entero filtrado → seq_len; C2 prod = chunks parciales (30–52 filas) → **no equivalente** salvo chunk ~8 s denso y misma cadena preprocess.

### 2.4 Preprocess Triton

10. **Sí** en intención: 640×360, letterbox 640, BGR→RGB, 8 KP 5–12, [0,1], Triton, umbrales 0.5 / 5 KP con `export_iamodel_full`.
11. Offline: **1 infer / frame decodificado** (puede 15 fps). C1: RTSP ~12 Hz + hold-last. Train recibe **seq_len tras resize sobre válidos**, no “8×12” explícito. **Paridad = mismo MP4 + track + densidad infer.**

### 2.5 `user_*`

12. **`user_{track_id}`** (ByteTrack). `single_user_only` → un user o skip. ID change mid-clip → varios `user_*` o skip.
13. **`valid_mask.npy` = frame_valid (≥5 KP). `poses.npy` = `poses_full[valid_mask]`** en disco.

### 2.6 `clip_buffer` producción

14. **Gold:** `triton_pose_extract` sobre **`clip_buffer.mp4`** descargado (mismo `pose_bundle_id`) → **`user_*/poses_full`** valida **`merged.npy`** (no chunk suelto para entrenamiento).
15. Mantener **merged + clip_buffer** para replay. **`merged` = concat(chunk_*)** mismas reglas offline; **`merged_valid_mask` desde valid_mask por chunk** (bugs cola oct-2026). Para train/paridad 41|48: clips offline desde `clip_buffer` **o** artefacto clip-equivalente por sesión; **no entrenar con chunks sueltos 2–4 s**.

**Script misma cadena preprocess train:** `experiments/training/campaign/score_pose_ensemble.py` (repo IAModel).

---

## 3. Golden clip

**Estado:** pendiente adjuntar ~6–8 s, 1 persona.

```bash
cd IAModel/experiments/triton_pose_extract && ./run_extract.sh extract --video ... --out ... --validate
```

Informe: `poses_full.shape`, `valid_mask.sum()`, T/duration, `meta.frame_count`.

**Frase TCN (IAModel):** tras loader, tensor **`[1, seq_len, features]`** con **seq_len=96** (modelo 41) o **64** (48), **features=32** (8 KP × x,y,dx,dy tras velocity).

**Contraste prod (no gold):** export **d9a71** — [`C1_CONTINUOUS_EXPORT_TRAIN_PARITY_HANDOFF.md`](./C1_CONTINUOUS_EXPORT_TRAIN_PARITY_HANDOFF.md).

**Enlace golden:**

```
(pegar aquí)
```

---

## 4. Evidencia C1 piloto (oct 2026)

| session_id (prefix) | Artefacto | T full | source_frames | all-NaN | valid_ratio | span (s) | Nota |
|---------------------|-----------|--------|---------------|---------|-------------|----------|------|
| `6ce8795…` | chunk_001 | 85 | 85 | 1/85 | ~0,99 | ~7,96 | Casi 1:1 ingest/export |
| `4affc3dd…` | chunk_001 | 89 | 89 | 40/89 | ~0,51 | ~7,34 | Rejilla OK; muchas filas no IAModel-válidas |
| `4c39a7ec…` | chunk_001 | 96 | 102 | 0/96 | ~0,49 | ~7,55 | Recorte **96 (=8×12)**; pasos delta≈0 |

Validar siempre **`merged.npy`** vs re-extract offline, no solo chunk_001.

---

## 5. Checklist IAModel

| # | Afirmación | Sí / No |
|---|------------|---------|
| A | `user_*` offline ≈ `merged` C1 | **Sí** (validación numérica; arreglar `merged_valid_mask`) |
| B | T = frames clip @ 12 fps siempre | **No** (T = frames decodificados; 12 solo si MP4 12 CFR) |
| C | Huecos NaN, no omitir filas | **Sí** con `keep_all`; **No** con `frame_policy: iamodel` |
| D | Train ventana 96 @ 12 Hz | **No** (seq_len modelo tras filter `valid_mask` sobre clip entero) |
| E | `poses.npy` = `poses_full[valid_mask]` | **Sí** |
| F | Re-extract `clip_buffer` = gold C1 | **Sí** |

**Comentarios IAModel:** chunks C1 ≠ unidades entrenamiento; piloto oct-2026 chunks &lt;8 s; `merged_valid_mask` incoherente en cola en algunos exports; paridad Triton OK, timeline/máscaras/scoring pendiente.

Referencias IAModel: `docs/C1_CONTINUOUS_EXPORT_TRAIN_PARITY_HANDOFF.md`, `docs/AGENT_TRITON_NPY_BRIEF.md`.

---

## 6. Historial

### Iteración 1 — 2026-10-19

Respuesta oficial IAModel pegada en [Anexo A](#anexo-a--respuesta-iamodel-2026-10-19). Resumen integrado en §1–§5.

### Iteración 2 — 2026-10-19 (C1: §10.2–§10.8; IAModel: aceptación)

**Estado IAModel:** **aceptado** — contrato §10 Nivel 1; **sin cambio de loader / `PoseDataset`** para cerrar Nivel 1 (batch y 41|48 siguen clip `user_*` + filter + resize).

**Aceptación IAModel (confirmada):**

> Adoptamos **kanvis_pose_v1 Nivel 1** (§10.1). Golden según **§10.2–§10.3** (ε **1e-4**, §8.3). **C2** mantiene guardrails de ventana mínima (**§10.4**); IAModel no los sustituye. **Campaign2 / train** = solo **`user_*` offline** (**§10.7–§10.8**). Piloto: **`clip_buffer` 12 fps CFR** como referencia gold **merged**; train histórico heterogéneo **sin re-encode masivo** (**§10.5**). Sujeto gold: **`single_user_only=true`** en piloto 1 persona; multi-track según **§10.6**.

**Script gold compartido (IAModel):**

```bash
cd IAModel/experiments/triton_pose_extract
./run_extract.sh extract --video /path/to/clip_buffer.mp4 --out /tmp/gold_offline --validate
python3 compare_c1_golden.py \
  --c1-session /path/to/session_pose_export \
  --offline-clip /tmp/gold_offline/data_result/.../clip_name \
  --chunk chunk_001 --json /tmp/gold_report.json
```

- Test **merged ↔ offline** clip entero (§8.1–§8.4).
- Test **chunk ↔ slice** por `pose_frame_timestamps` + `playback_start_at` (§10.2).

Detalle normativo: **§10.2–§10.8** (abajo).

---

## 7. Referencias C1

- [`POSE_EXPORT_IAMODEL.md`](./POSE_EXPORT_IAMODEL.md)
- [`MENSAJE_IAMODEL_TRITON_CLIP_EXTRACTOR_RESPUESTA.md`](./MENSAJE_IAMODEL_TRITON_CLIP_EXTRACTOR_RESPUESTA.md)
- [`C1_CONTINUOUS_EXPORT_TRAIN_PARITY_HANDOFF.md`](./C1_CONTINUOUS_EXPORT_TRAIN_PARITY_HANDOFF.md)

---

## 8. Plan C1 (derivado de respuesta IAModel)

### Objetivo de paridad real

**`merged.npy` + `merged_valid_mask.npy`** deben poder compararse con **`user_*/poses_full.npy`** tras **re-extract offline** del mismo **`clip_buffer.mp4`** (mismo bundle, `keep_all`, sin hold-last en export).

No basta “12 Hz en RTSP”: offline es **1 infer / frame del MP4**. Si el `clip_buffer` es **12 fps CFR**, objetivo:

**`merged.shape[0] ≈ meta.frame_count` del `clip_buffer`** (y timestamps coherentes).

### Acciones

| P | Acción | Estado |
|---|--------|--------|
| P0 | Runbook: MinIO `clip_buffer` + `merged*` → `triton_pose_extract` → diff numérico vs `merged` | Pendiente |
| P0 | **`merged_valid_mask`**: solo propagar `valid_mask.npy` de chunks; auditar cola (handoff d9a71) | Parcial en repo |
| P1 | Ingest/export timeline: `pose_timeline_sequence_id` (no RTSP 15 Hz) | Hecho (desplegar) |
| P1 | **No downsample 102→96** en chunks &lt;8 s; evitar coords repetidas con `valid=False` (hold-last export) | Pendiente |
| P1 | **`clip_buffer` 12 fps CFR** documentado; alinear expectativa **T_merged** con **frame_count vídeo**, no solo `12×duration` | Verificar encode sesión |
| P2 | **`merged_keypoint_mask.npy`** (concat) | Pendiente |
| P2 | Tier L single-person defer | Hecho (desplegar) |
| P3 | C2: no tratar chunk 2–4 s como unidad train; scoring alineado a `score_pose_ensemble.py` / ventana densa | Doc C2 / IAModel |

### Criterio de aceptación

1. Re-extract offline del `clip_buffer` de la sesión.
2. **`len(merged) == len(user poses_full)`** ± tolerancia acordada (ideal: igual) — gold **merged ↔ clip entero**; gold **chunk ↔ slice temporal** (§10.2).
3. **Coords:** en filas alineadas con **ambos** `valid_mask=True`, **max |Δ| &lt; ε = 1e-4** en coords **[0, 1]** (por KP, eje x/y). **Máscara:** en las mismas filas alineadas, **`valid_mask` idéntico** (True/False); filas solo-NaN deben coincidir en presencia de NaN (no comparar coords en filas inválidas).
4. **`merged_valid_mask[i]` == offline `valid_mask[i]`** en toda la cola (sin True fantasma).

**Alineación temporal (tolerancia índice):** emparejar filas chunk vs offline por **`pose_frame_timestamps.npy`** con **|Δt| ≤ 0,5 / fps_ref**, con `fps_ref` = `meta.json` sesión `pose_fps` o `ffprobe` del `clip_buffer` (§10.2).

### Qué no confundir

- **Chunks MinIO** → streaming / C2 ventanas; **no** sustituto del npy de entrenamiento.
- **Train TCN:** filtra válidos → resize a **64/96**; **C2 en chunk corto** no replica eso sin clip completo o ventana ~8 s densa.

---

## 9. Paridad **exacta** con IAModel (estrategia C1)

Tu objetivo: **mismos `.npy` que entrenamiento** → mejor comportamiento del modelo en producción.

| Enfoque | ¿Igual que IAModel? | Rol |
|---------|---------------------|-----|
| RTSP ~12 Hz → chunks → **concat → merged** | **No** (densidad, hold-last, tracker continuo) | Tiempo real / C2 |
| **`clip_buffer.mp4` → 1 infer / frame → merged** | **Sí** (mismo contrato que ClipExtractor) | Train / eval / gold |

**Reglas IAModel a respetar en `merged` (y en chunks si aplica):**

- **T = número de frames del MP4** (100 % decodificados), `keep_all` → NaN + `valid_mask=False` si no hay pose válida.
- **Sin hold-last** en `poses_full` exportado; filas inválidas = **NaN**, no coords repetidas.
- **`valid_mask`**: ≥5 KP visibles, conf export 0.5; **`poses.npy` = válidos**.
- **`keypoint_mask.npy`** por keypoint.

**En repo (kanvis-monitoring):**

| Pieza | Estado |
|-------|--------|
| `pose/clip_buffer_iamodel_export.py` | Módulo: decode MP4 + arrays IAModel full |
| `pose_merged_export_from_clip_buffer` (yaml) | Flag; **hook al cerrar sesión** (sustituir merge por chunks) — **pendiente cableado** |
| Chunks parciales: **no recortar a 96** si span &lt; ~87,5 % ventana 8 s | Hecho |
| Export IAModel: **NaN en fila si `valid_mask=False`** | Hecho |
| Rejilla ingest 12 Hz (`pose_timeline_sequence_id`) | Hecho (mejora chunks; merged gold sigue siendo clip) |

**Validación:** re-extract offline del mismo `clip_buffer` vs `merged.npy` (§8 criterios).

---

## 10. Contrato común (C1 + IAModel + C2) — propuesta acordada

**Objetivo:** una sola definición de “cómo es un `poses_full.npy` válido”, tanto en **`user_*/`** (batch) como en **`chunk_*/`** (MinIO / robo).

**Ancla recomendada:** reglas ya usadas en entrenamiento (**ClipExtractor + `export_iamodel_full` + `PoseDataset`**). C1 adopta ese perfil en MinIO; IAModel **no** cambia el loader salvo acuerdo explícito. C2/Campaign2 consume arrays que cumplan el contrato (no reinterpreta umbrales distintos).

**Lectura única (formato vs duración):** **Paridad de contenido** = Nivel 1 (misma semántica por fila). **Paridad de distribución temporal** (T, densidad fps) se valida con gold **`clip_buffer` + merged** o slice offline; **no** se exige que un chunk RTSP de 3 s tenga la misma T que un clip de entrenamiento de 10 s.

### Nivel 1 — Semántica de arrays (obligatorio prod + train)

| Regla | Valor |
|-------|--------|
| Shape | `(T, 8, 2)` float32, KP COCO **5–12**, coords **[0, 1]** |
| Conf export | NaN en KP si conf ≤ **0,5** |
| `valid_mask` | True si ≥ **5** KP no-NaN; False en `is_synthetic` |
| `poses.npy` | `poses_full[valid_mask]` |
| `keypoint_mask.npy` | Coherente con NaN post-umbral |
| Gap-fill en `poses_full` | **`none`** (sin interpolate / hold-last en export) |
| `pose_bundle.json` | Mismo `pose_bundle_id`, plan/pt, tracker YAML |

**C1:** `pose_export_iamodel_full: true` (env `CONTINUOUS_POSE_EXPORT_IAMODEL_FULL=true` en piloto).  
**IAModel:** mantener `config/pose.yaml` alineado al mismo bundle y umbrales.  
**C2:** preprocess = misma cadena que `score_pose_ensemble.py` / `PoseDataset` sobre lo leído del chunk.

### Nivel 2 — Una fila = un frame inferido (paridad numérica fuerte)

| Modo | Definición de **T** | Uso |
|------|---------------------|-----|
| **MP4 dense** (IAModel batch) | T = **todos** los frames BGR decodificados del MP4 (`keep_all`) | Entrenamiento, gold, re-extract |
| **Timeline ingest** (C1 RTSP ~12 Hz) | T = frames **ingeridos** en la ventana del chunk (rejilla sesión, no 15 Hz RTSP crudo) | Tiempo real; **aproxima** MP4 dense si fps estable |

Para que **`chunk_*/poses_full.npy`** coincida **numéricamente** con “IAModel sobre los mismos píxeles” hace falta **Nivel 2 MP4 dense por ventana**: al publicar chunk (o al cerrar), re-inferir **cada frame JPEG/MP4 del chunk** con la misma pipeline que ClipExtractor — no solo re-exportar la cola RTSP. Opcional futuro en C1; hoy Nivel 1 + ingest 12 Hz.

### Qué **no** unificar (distinto propósito)

- **Unidad de tiempo:** train suele usar **clip entero**; C2 usa **chunks** (~3–8 s) y acumulación — OK si cada chunk cumple Nivel 1.
- **`single_user_only`:** batch IAModel descarta multi-track; C1 usa sesiones/ROI — política de producto distinta, no del formato `.npy`.
- **Gates de apertura de sesión** (conf 0,35 / 2 KP): pueden seguir siendo permisivos; **solo el upload MinIO** debe ser perfil IAModel full.

### Checklist de adopción

| Paso | Responsable | Hecho |
|------|-------------|-------|
| Activar IAModel full en piloto `.env.docker` | C1 ops | Pendiente |
| Ver `chunk_*/meta.json` → `iamodel_full: true` | C1 | Tras deploy |
| Golden: `compare_c1_golden.py` + re-extract `clip_buffer` (§10.2) | IAModel + C1 | Script en repo IAModel; P0 runbook pendiente en piloto |
| Documentar versión contrato en `pose_bundle.json` (`contract_version: kanvis_pose_v1`) | Ambos | §10.6 |

### 10.2 Golden — alinear filas `chunk_*` ↔ frames `clip_buffer`

**Entrada MinIO (por sesión piloto gold):**

- `clip_buffer.mp4`
- `meta.json` sesión: `playback_start_at`, `pose_timeline_start_at`, `pose_fps`, `track_id` (sujeto sesión)
- `chunk_NNN/pose_frame_timestamps.npy` (**`utc_epoch_seconds`**, ver `chunk_NNN/meta.json` → `pose_frame_timestamps_unit`)
- `chunk_NNN/poses_full.npy`, `chunk_NNN/valid_mask.npy`

**Offline (IAModel):** ClipExtractor sobre **`clip_buffer.mp4` completo**, `keep_all`, perfil IAModel full, **`single_user_only=true`** si sesión piloto **1 persona** (`max_persons_in_frame=1`); si skip multi-track, excluir sesión del gold o acordar `single_user_only=false` + track dominante (§10.5).

**Regla de índice vídeo (por fila i del chunk):**

1. `t_i = pose_frame_timestamps[i]` (epoch UTC, segundos).
2. `t0_playback = epoch(playback_start_at)` desde `meta.json` sesión (ISO UTC).
3. **`video_time_s_i = t_i - t0_playback`** (mismo criterio que `frontend_playback.json` → `video_time_s ≈ pose_timeline_time_s + pose_offset_seconds`; el offset queda absorbido en `playback_start_at` vs timeline pose).
4. `fps_ref = pose_fps` sesión, o `ffprobe` del `clip_buffer` si falta.
5. **`k_i = round(video_time_s_i * fps_ref)`**, clamp `[0, frame_count-1]`.
6. Offline: fila **`user_*/poses_full[k_i]`** (mismo `user_*` elegido en §10.5) vs **`chunk_NNN/poses_full[i]`**.

**No usar** como gold primario: `merged[t0:t1]` sin alinear por timestamp (concat dedupe puede desplazar índices). **Sí usar** merged ↔ offline **solo** en test **clip entero** (§8 criterios 1–2).

**Sesiones gold piloto:** preferir **1 track / 1 persona**; multi-persona = otro protocolo (ReID / track dominante).

### 10.3 ε y comparación (ref. §8.3)

| Qué | Regla |
|-----|--------|
| ε coords | **1e-4** en [0, 1], max por KP (x,y) |
| Filas | Solo donde **ambos** `valid_mask=True` tras alinear por §10.2 |
| NaN / inválidas | **`valid_mask` igual** fila a fila; coords no comparadas si cualquier lado False |
| Δt alineación | **≤ 0,5 / fps_ref** s entre `t_i` y timestamp implícito del frame `k_i` offline |

### 10.4 C2 — chunks cortos y T′ tras filtrar

§10 Nivel 1 **no obliga** a que C2 puntúe siempre. Política acordada C1 ↔ C2 (piloto; repo `c2_inference_service`):

| Condición | Acción C2 |
|-----------|-----------|
| Ventana acumulada de pose **válida** &lt; **~8 s** (`min_window_seconds`) | **No alarmar** / no stage-1 decisivo (OK técnico, sin score de robo fiable) |
| `is_last_chunk` y `mono_span_ms` &lt; **2000** | Ignorar para alarma |
| `quality_pass=false` o `valid_mask_true_ratio` &lt; **0,5** (chunk meta) | No alarmar |
| Antes de `KANVIS_MIN_INFER_CHUNKS_BEFORE_DECISION` | Calentamiento (sin decisión C3) |

Tras `poses_full[valid_mask]`, **T′ pequeño** + `temporal_resize` sigue siendo matemáticamente válido pero **no debe disparar robo** hasta ventana densa — eso explica FP en piloto aunque Nivel 1 esté bien; Fase 0 guardrails en [`PLAN_CORRECCION_EXPORT_CHUNKS_C1.md`](./PLAN_CORRECCION_EXPORT_CHUNKS_C1.md).

### 10.5 `clip_buffer` 12 fps CFR vs train heterogéneo

| Ámbito | Acuerdo |
|--------|---------|
| **Piloto prod C1** | `clip_buffer` **12 fps CFR** → gold merged: **`T_merged ≈ frame_count`** del buffer (checklist B sigue siendo “T = frames decodificados”, no “12 siempre”). |
| **Train histórico** | Clips 15/25/30 fps **sin re-encode masivo** por este contrato. |
| **Implicación modelo** | Nivel 1 alinea **semántica**; densidad temporal train vs infer puede diferir — mitigación: loader `temporal_resize` + eval gold 12 fps piloto; **re-encode dataset futuro** = decisión producto aparte. |

### 10.6 Sujeto: `merged` / chunk vs `user_*` offline

| C1 | Offline gold |
|----|----------------|
| Un chunk = **un sujeto de sesión** (`chunk.meta` / `frames_meta` → **`track_id`** ByteTrack en vivo) | **`user_{id}`** = track ByteTrack **del clip** (IDs **no** comparables numéricamente entre tubos) |
| `merged` = timeline del **mismo sujeto de sesión** (poses ingeridas para ese `track_id`) | Comparar por **tiempo + geometría**, no por igualdad `track_id` C1 == carpeta `user_3` |

**Gold piloto (1 persona):** offline `single_user_only=true` → un solo `user_*`; alinear §10.2. **Multi-track:** offline `single_user_only=false` + elegir `user_*` con **mayor solapamiento temporal** con `[t_0, t_{T-1}]` del chunk (o mayor `valid_mask.sum()` en el intervalo); C1 fija sujeto en **`meta.json` sesión `track_id`** + `person_id` si existe.

### 10.7 `contract_version: kanvis_pose_v1` — campos mínimos

**`pose_bundle.json` (chunk y sesión):**

- `contract_version`: `"kanvis_pose_v1"`
- `pose_bundle_id`, SHA plan/pt, tracker YAML, backend Triton, umbrales export (**0,5 / 5 KP**)

**`chunk_NNN/meta.json` (marker):**

- `iamodel_full`, `export_min_kp_conf`, `export_min_visible_kps`
- `pose_chunk_gap_fill`, `pose_chunk_gap_fill_applied`
- `pose_frame_timestamps_unit` (= `utc_epoch_seconds`)
- `pose_target_fps`, `source_frames_count`, `poses_full_shape`
- `pose_bundle_id`, `pose_bundle_ref`
- `track_id`, `session_id`, `chunk_id`, `chunk_timeline_span_seconds`, `valid_mask_true_ratio`

**Extract batch IAModel (`meta.json` clip):** mismo `contract_version` + `pose_bundle_id` + fps/frame_count para scripts diff compartidos.

### 10.8 Campaign2 / entrenamiento vs MinIO

| Fuente | Uso |
|--------|-----|
| **`user_*/poses_full` offline** (ClipExtractor) | **Dataset Campaign2 / entrenamiento TCN** |
| **`chunk_*/poses_full` MinIO** | **Inferencia C2 en prod**, eval piloto, regresión §10.2 |
| **`chunk_*` → train directo** | **No** salvo re-extract explícito desde `clip_buffer` a unidad clip |

Eval offline “8 s @ 12 Hz” en pipeline Campaign2 **permanece** en su repo; no sustituye §10.2 chunk gold.

---

## Anexo A — Respuesta IAModel (2026-10-19)

```
=== RESPUESTA IAModel — Paridad poses_full (2026-10-19) ===

§1 — ¿user_*/poses_full offline ≡ merged.npy C1?

Sí, con matices. Para comparar paridad de contenido temporal del sujeto en el MP4 de referencia:
- Equivalente offline: user_<track>/poses_full.npy sobre TODO el clip (mismo clip_buffer.mp4, mismo pose_bundle_id, mismas reglas IAModel full).
- merged.npy C1 debería ser esa misma timeline (concat temporal de chunks con mismas reglas).
- chunk_NNN/poses_full.npy NO es 1:1 con el npy offline de entrenamiento: es ventana parcial (objetivo ~8 s @ 12 Hz; en piloto oct-2026 vimos chunks de 2–4 s, micro-chunks de <1 s — sesión d9a71, ver C1_CONTINUOUS_EXPORT_TRAIN_PARITY_HANDOFF.md).
- Entrenamiento TCN 41|48: un ejemplo = un clip completo (data_result/{cat}/{clip}/user_*), NO un chunk MinIO. C2 prod puntúa por chunk/ventana clip corta → desalineación de inferencia aunque Triton sea el mismo.

No equivaler chunk_* ↔ user_* sin recortar/agregar. Sí equivaler merged ↔ user_* tras validación numérica (shape, T, timestamps, valid_mask).

§2.1 — Filas T y fps

1) Offline (triton_pose_extract / ClipExtractor / c1_like): T = una fila de pose por cada frame BGR decodificado del MP4 (inferencia en 100 % de frames decodificados). No re-muestreamos a 12 Hz en batch salvo que el MP4 ya sea 12 fps CFR.

2) Tras CSV+FFmpeg: suele cumplirse frame_count == video_frame_count == len(poses_full) por user_* exportado. Si el MP4 es 15/25/30 fps, T refleja esa fps, no 12 fijo.

3) pose_frame_timestamps.npy: en C1/MinIO sí (epoch UTC). En batch offline clásico a menudo NO se escribe (solo meta.json fps/frame_count). Si existe offline: frame_idx/fps del clip, no rejilla 12 Hz artificial.

Contraste C1: ingest ~12 Hz + hold-last + pose_resampled parcial → T y fps efectivo pueden no coincidir con offline “todos los frames del MP4”.

§2.2 — Huecos, NaN, keep_all

4) frame_policy keep_all (default): sin detección en frame de track → fila NaN en poses_full + valid_mask=False. Con frame_policy iamodel: si el track no aparece, NO se añade fila (T más corto) — no paridad con grid C1 full.

5) Offline keep_all: no hold-last; huecos = NaN (paridad deseada con pose_chunk_gap_fill: none). En prod vimos cola con coords repetidas + valid_mask=False (d9a71 chunk_004) — NO contrato offline.

6) Offline: no is_synthetic en npy histórico. valid_mask = ≥5 KP visibles, conf ≥0.5 export. C1 synthetic debe excluirse igual en valid_mask.

§2.3 — Loader entrenamiento (41|48)

7) PoseDataset / campaña pose_source full: lee poses_full.npy en user_*.

8) Antes del forward: poses = poses_full[valid_mask] (elimina filas inválidas), luego normalize → add_velocity → temporal_resize(seq_len). seq_len del checkpoint (bin_full: ~96 exp 41, ~64 exp 48).

9) NO es ventana fija 8 s = 96 filas. Es T' = valid_mask.sum() del clip entero, luego resize a seq_len fijo. “8 s @ 12 Hz” es doc C1 continuo, no regla exacta del loader.

Implicación: train = clip entero filtrado → seq_len 64/96; C2 prod = chunks parciales (p.ej. 30–52 filas) source_window clip → no equivalente salvo chunk ~8 s dense y misma cadena preprocess.

§2.4 — Preprocess Triton

10) Sí en intención: 640×360, letterbox 640, BGR→RGB, 8 KP COCO 5–12, coords [0,1], Triton end2end, umbrales 0.5 / 5 KP con export_iamodel_full alineado.

11) Offline: 1 infer / frame decodificado (puede 15 fps). C1 RTSP ~12 Hz + hold-last. El modelo en train recibe seq_len tras resize sobre secuencia válida del clip, no “8×12” explícito. Paridad = mismo MP4 + track + densidad infer; no basta mismo Triton.

§2.5 — user_*

12) Carpetas user_{track_id} (ByteTrack). single_user_only true (default batch): un user o skip clip. export_primary_user_on_multi_track: track dominante. ID change mid-clip → varios user_* o skip.

13) valid_mask.npy = frame_valid (≥5 KP). poses.npy = poses_full[valid_mask] en disco.

§2.6 — clip_buffer producción

14) Sí: gold = triton_pose_extract sobre clip_buffer.mp4 descargado (mismo pose_bundle_id) → user_*/poses_full valida merged.npy (no chunk suelto para entrenamiento).

15) Recomendación:
- Mantener merged + clip_buffer para replay.
- merged = concat(chunk_*) con mismas reglas que offline; merged_valid_mask debe derivar de valid_mask por chunk (hoy hay bugs: cola chunk_004 invalid pero merged_valid_mask True — handoff oct-2026).
- Para train/paridad 41|48: generar clips offline desde clip_buffer (1 npy/clip) O artefacto clip-equivalent validado por sesión; no entrenar con chunks sueltos de 2–4 s.

Script score mismo preprocess train: experiments/training/campaign/score_pose_ensemble.py

§3 — Golden clip

Pendiente adjuntar ~6–8 s, 1 persona:
  cd IAModel/experiments/triton_pose_extract && ./run_extract.sh extract --video ... --out ... --validate
Informe: poses_full.shape, valid_mask.sum(), T/duration, meta.frame_count.
Frase TCN: tras loader, tensor [1, seq_len, features] con seq_len=96 (modelo 41) o 64 (modelo 48), features=8*4=32 (8 KP x,y,dx,dy tras velocity).

Contraste prod (no gold): export d9a71 — C1_CONTINUOUS_EXPORT_TRAIN_PARITY_HANDOFF.md

§5 — Checklist

A user_* offline ≈ merged C1 → Sí (validar numéricamente; arreglar merged_valid_mask)
B T = frames clip @ 12 fps siempre → No (T = frames decodificados; 12 solo si MP4 12 CFR)
C Huecos NaN, no omitir filas → Sí con keep_all; No con frame_policy iamodel
D Train ventana 96 @ 12 Hz → No (seq_len modelo tras valid_mask filter sobre clip entero)
E poses.npy = poses_full[valid_mask] → Sí
F Re-extract clip_buffer = gold C1 → Sí

Comentarios: chunks C1 ≠ unidades entrenamiento; piloto oct-2026 chunks <8 s, merged_valid_mask incoherente en cola; paridad Triton OK, timeline/máscaras/scoring pendiente.

Referencias IAModel: docs/C1_CONTINUOUS_EXPORT_TRAIN_PARITY_HANDOFF.md, docs/AGENT_TRITON_NPY_BRIEF.md
```
