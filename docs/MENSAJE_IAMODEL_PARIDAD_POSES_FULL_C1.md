# Paridad `poses_full.npy` — C1 continuo (MinIO) vs extracción offline (ClipExtractor)

**Para:** agente / equipo IAModel — extracción Triton por clip (`kanvis-triton-pose`, ClipExtractor, `PoseDataset`)  
**De:** equipo C1 — `kanvis-monitoring` (piloto wudalkot)  
**Objetivo:** que los arrays de pose de producción sean **intercambiables en contenido** con los del entrenamiento, aunque la **jerarquía en disco** sea distinta.

---

## 1. Mapeo de almacenamiento (mismo clip, distinto árbol)

| Offline (batch por clip) | C1 continuo (MinIO sesión) | Notas |
|--------------------------|----------------------------|--------|
| `clip.mp4` (~12 fps CFR) | `clip_buffer.mp4` (+ pre/post según buffer) | Mismo rol: vídeo de referencia |
| `meta.json` | `meta.json` (sesión) + `chunk_NNN/meta.json` | C1 parte en ventanas ~8 s |
| `pose_bundle.json` | `chunk_NNN/pose_bundle.json` (y manifiesto sesión) | Mismo bundle / SHA plan |
| `user_<id>/poses_full.npy` | **`merged.npy`** (sesión) | **Equivalente semántico** (timeline completa del sujeto en el clip) |
| `user_<id>/valid_mask.npy` | **`merged_valid_mask.npy`** | |
| `user_<id>/poses.npy` | Concat / filtro por chunks o derivado de merged | `poses_full[valid_mask]` en IAModel full |
| `user_<id>/keypoint_mask.npy` | Por chunk `keypoint_mask.npy` | |
| *(no hay chunks)* | `chunk_NNN/poses_full.npy` | Ventana parcial (~8 s stride); **no** es 1:1 con `user_*/poses_full` del offline |

**Hipótesis de trabajo:** para el **mismo** `clip.mp4` / `clip_buffer.mp4` y mismo track/sujeto, los **`.npy` deberían coincidir en reglas** (shape, fps, NaN, valid_mask), salvo diferencias de tracker continuo vs por-clip y de recorte temporal.

**Pregunta de cierre:** ¿confirmáis que **`user_*/poses_full.npy` offline ≡ `merged.npy` C1** (no `chunk_*/poses_full.npy`) para comparar paridad?

**Respuesta IAModel:**

**Sí, con matices.** Para paridad de **contenido temporal del sujeto en el MP4 de referencia**, el equivalente offline es **`user_<track>/poses_full.npy` sobre todo el clip**, no un `chunk_NNN` aislado.

- **`merged.npy` C1** debería ser la concatenación temporal de la misma secuencia que un **`user_*/poses_full`** offline ejecutado sobre el **mismo** `clip_buffer.mp4` (mismo track, mismo `pose_bundle_id`, mismas reglas IAModel full).
- **`chunk_NNN/poses_full.npy` no es 1:1** con el npy offline: es una **ventana parcial** (ideal ~8 s @ 12 Hz, pero en piloto oct-2026 vimos chunks **mucho más cortos** y micro-chunks; ver `C1_CONTINUOUS_EXPORT_TRAIN_PARITY_HANDOFF.md`, sesión `d9a71`).
- **Entrenamiento (TCN 41|48)** en campaña usa **un ejemplo = un clip completo** (carpeta `data_result/{cat}/{clip}/user_*`), no un chunk MinIO. **C2 en prod puntúa por chunk/clip corto** → desalineación de inferencia aunque Triton sea el mismo.

**No equivaler** `chunk_*` ↔ `user_*` sin recortar/agregar; **sí equivaler** `merged` ↔ `user_*` **tras validación numérica** (shape, T, timestamps, valid_mask).

---

## 2. Preguntas concretas (rejilla temporal y loader)

### 2.1 Filas T y fps

1. En offline, **`poses_full.shape[0]` (T)** = ¿**cada frame decodificado** del `clip.mp4`, o una rejilla **12 Hz** aunque el MP4 fuente fuera 15/24/30 fps antes del CFR?
2. Tras exportar **`clip.mp4` a 12 fps CFR**, ¿siempre **`T == meta.frame_count == meta.video_frame_count`** (1:1 vídeo ↔ filas)?
3. ¿**`pose_frame_timestamps.npy`** (si existe offline) es uniforme `t0 + i/12`, o PTS real del MP4?

**Respuesta:**

1. **Offline (`triton_pose_extract` / `ClipExtractor` / `c1_like`):** **T = un frame de pose por cada frame BGR decodificado del MP4** (`read_frames_bgr` → inferencia en **100 %** de esos frames). **No** re-muestreamos a 12 Hz en el export batch salvo que el **propio MP4** ya venga a 12 fps CFR.
2. **Tras recorte CSV + FFmpeg:** en meta suele **`frame_count == video_frame_count == len(poses_full)`** para cada `user_*` exportado (1:1 con frames procesados). Si el MP4 fuente es 15/25/30 fps, **T refleja esa fps**, no 12 fijo.
3. **`pose_frame_timestamps.npy`:** en **MinIO/C1** sí (epoch UTC). En **batch offline clásico** a menudo **no** se escribe en disco (solo `meta.json` con `fps` / `frame_count`). Cuando exista, offline histórico = **`frame_idx / fps`** del clip, no necesariamente rejilla 12 Hz artificial.

**Contraste C1:** ingest ~12 Hz + hold-last + resample parcial hacia 96 (`pose_resampled`) → **T y fps efectivo pueden no coincidir** con offline “todos los frames del MP4”.

### 2.2 Huecos, NaN y `frame_policy: keep_all`

4. Con **`keep_all`**, cuando no hay detección o track: ¿**fila en `poses_full` con NaN** + **`valid_mask=False`**, u **omitís** la fila (T más corto)?
5. ¿Hay **hold-last / interpolate** en coords de `poses_full` offline, o solo NaN en huecos (paridad con C1 `pose_chunk_gap_fill: none`)?
6. ¿**`is_synthetic`** o equivalente existe offline? ¿Afecta a `valid_mask` como en C1 (`export_iamodel_full`)?

**Respuesta:**

4. **`frame_policy: keep_all` (default en `config/pose.yaml` vía `extract_stack`):** si el track existe pero **no hay detección en ese frame**, se añade fila con **NaN** en `poses_full` + **`valid_mask=False`** (vía `keypoint_mask` / reglas export). Si **`frame_policy: iamodel`**, cuando el track **desaparece del frame no se añade fila** (T más corto) — **no** es paridad con C1 full grid.
5. **Offline batch (keep_all):** **no hold-last** en coords; huecos = **NaN** (paridad deseada con C1 `pose_chunk_gap_fill: none`). **Excepción observada en prod:** cola con coords **repetidas** y `valid_mask=False` (`d9a71` `chunk_004`) — **no** es el contrato offline.
6. **Offline:** no hay flag `is_synthetic` en npy histórico; **`valid_mask`** = criterio IAModel (**≥5 KP** visibles, conf ≥ 0.5 en export). C1 puede marcar synthetic en ingest; debe **excluirse** de `valid_mask` igual que en offline.

### 2.3 Qué lee el entrenamiento (TCN / motion 41|48)

7. ¿El loader (`PoseDataset` u otro) consume **`poses_full.npy`**, **`poses.npy`**, o **`poses_full[valid_mask]`**?
8. ¿Las filas con **`valid_mask=False`** entran al modelo (con NaN/máscara) o se **eliminan** antes de la ventana?
9. Ventana de entrenamiento: ¿**8 s fijos @ 12 Hz (= 96 filas)** siempre, o **T = duración del clip**?

**Respuesta:**

7. **`PoseDataset` / campaña (`train_model_operations`, `pose_source: full`):** lee **`poses_full.npy`** desde `user_*`.
8. **Antes del forward:** aplica **`poses = poses_full[valid_mask]`** (elimina filas inválidas), luego `normalize_sequence` → `add_velocity` → **`temporal_resize(poses, seq_len)`** (seq_len del checkpoint: p. ej. **96** exp 41, **64** exp 48 en `bin_full`).
9. **Ventana:** **no** es “8 s fijos = 96 filas” en el loader. Es **T′ = valid_mask.sum()** tras filtrar, luego **resize a seq_len fijo** del modelo. Clips cortos/largos → **misma seq_len** (interpolación/subsampling). La noción “8 s @ 12 Hz” es **documentación C1 continuo**, no la regla exacta del loader offline.

**Implicación:** entrenáis con **secuencia del clip entero (filtrada)** comprimida a **64/96**; C2 prod suele ver **chunks parciales** (30–52 filas) con **`source_window: clip`** — **no equivalente** a un forward de entrenamiento salvo que el chunk sea ~8 s densos y se use la misma cadena.

### 2.4 Preprocess y Triton (paridad con C1)

10. Confirmar: **640×360**, letterbox 640, BGR→RGB, **8 KP COCO 5–12**, coords **normalizadas [0,1]** — ¿misma especificación que el bridge C1 → Triton?
11. Offline: ¿inferencia en **100 % frames** del clip vs C1 ~**12 Hz** en RTSP — ¿el modelo entrenado asume **densidad por frame de vídeo** o **12 muestras/s** independientes del vídeo?

**Respuesta:**

10. **Sí** en intención: letterbox 640, KP **COCO 5–12 → 8 coords**, normalizado **[0,1]**, Triton end2end, mismos umbrales export **0.5 / 5 KP** cuando `export_iamodel_full` está alineado con `pose_core.export`.
11. **Densidad:** offline batch = **1 infer / frame decodificado** (puede ser 15 fps si el clip es 15 fps). C1 RTSP ≈ **12 Hz infer** + hold-last entre inferencias. **El modelo no recibe “8 s × 12” explícito en train**; recibe **seq_len** tras resize sobre **toda la secuencia válida del clip**. Paridad real = **mismo MP4 + mismo track + misma densidad de infer**; no basta “mismo Triton”.

### 2.5 Tracker y carpetas `user_*`

12. Un clip con **1 persona** exportada: ¿siempre **`user_1/`** (o un solo `user_*`) con el track dominante? ¿Criterio si ByteTrack cambia ID mid-clip?
13. Umbrales **`min_kp_conf=0.5`**, **≥5 KP visibles** en export: ¿definen **exactamente** `valid_mask.npy` en disco?

**Respuesta:**

12. **`user_{track_id}/`** (ID ByteTrack numérico, no siempre `user_1`). Con **`single_user_only: true`** (default batch): **un solo user** o **skip** del clip. Con **`export_primary_user_on_multi_track`**: el track dominante. Cambio de ID mid-clip → **dos carpetas user_*** o skip según filtros.
13. **`valid_mask.npy` en disco** = frames que pasan **`frame_valid`** (≥5 KP visibles tras máscara conf). **`poses.npy`** = subconjunto **`poses_full[valid_mask]`** escrito en export.

### 2.6 Comparación con `clip_buffer` de producción

14. Si ejecutáis **ClipExtractor** sobre un **`clip_buffer.mp4`** descargado de MinIO (misma GPU/plan que piloto), ¿el **`user_*/poses_full.npy`** resultante es la **referencia gold** para validar C1 **`merged.npy`**?
15. ¿Recomendáis que C1 deje de publicar chunks intermedios para paridad y solo exporte **merged + clip** al estilo offline, o que **`merged` = concat(chunk_* )`** con las mismas reglas que un solo paso offline?

**Respuesta:**

14. **Sí — gold recomendado:** re-ejecutar **`triton_pose_extract`** (`./run_extract.sh extract --video clip_buffer.mp4`) con el **mismo `pose_bundle_id`** que piloto → **`user_*/poses_full.npy`** es referencia para validar **`merged.npy`** (no cada chunk suelto para entrenamiento).
15. **Recomendación IAModel:**
    - Mantener **`merged` + `clip_buffer`** para replay humano.
    - **`merged` debe ser `concat(chunk_* )` con mismas reglas** que un único paso offline **y** **`merged_valid_mask` = f(valid_mask por chunk)** — hoy hay **contradicciones** (ej. cola `chunk_004` inválida pero merged True; ver handoff oct-2026).
    - Para **entrenamiento / paridad 41|48**, o bien **generáis clips offline desde `clip_buffer`** (1 npy/clip), o bien C1 publica **un artefacto “clip-equivalent”** por sesión (= merged validado), **no** confiar en chunks sueltos de 2–4 s.

**Script local score (mismo preprocess que train):** `experiments/training/campaign/score_pose_ensemble.py` sobre `poses_full` + `valid_mask`.

---

## 3. Artefacto golden solicitado

Por favor adjuntar **un clip de referencia** (~6–8 s, 1 persona, pasillo), con:

```
<clip_name>/
  clip.mp4
  meta.json
  pose_bundle.json
  user_1/
    poses_full.npy
    valid_mask.npy
    poses.npy
    keypoint_mask.npy
    (pose_frame_timestamps.npy si aplica)
```

Y en texto:

- `poses_full.shape`, `valid_mask.sum()`, span temporal (s), **`T / duration`** ≈ fps efectivo.
- Una frase: **“El TCN recibe un tensor de shape … tras el loader”**.

**Enlace / ruta del golden:**

**Pendiente adjuntar** desde batch (6–8 s, 1 persona). Procedimiento:

```bash
cd IAModel/experiments/triton_pose_extract
./run_extract.sh extract --video /path/clip.mp4 --out /path/golden_out --validate
```

Informe mínimo por clip: `poses_full.shape`, `valid_mask.sum()`, `T/duration`, `meta.frame_count`.

**Contraste ya hecho (prod, no gold):** `session_pose_export_20261019_104309/d9a71…` — ver `C1_CONTINUOUS_EXPORT_TRAIN_PARITY_HANDOFF.md`.

---

## 4. Evidencia C1 piloto (oct 2026) — para contrastar

Tras alinear **`sequence_id` a rejilla ~12 Hz** en ingest (no contador RTSP 15 Hz):

| session_id (prefix) | Artefacto | T full | source_frames | all-NaN | valid_ratio | span (s) | Nota |
|---------------------|-----------|--------|---------------|---------|-------------|----------|------|
| `6ce8795…` | chunk_001 | 85 | 85 | 1/85 | ~0,99 | ~7,96 | Casi paridad 1:1 |
| `4affc3dd…` | chunk_001 | 89 | 89 | 40/89 | ~0,51 | ~7,34 | Rejilla OK; muchas filas no IAModel-válidas |
| `4c39a7ec…` | chunk_001 | 96 | 102 | 0/96 | ~0,49 | ~7,55 | Recorte a **96 (=8×12)**; ~50 % pasos delta≈0 |

**Pendiente C1 (según vuestras respuestas):** cap 96 en chunks &lt;8 s, ingest &gt;12 Hz, y validar **`merged.npy`** vs vuestro **`user_*/poses_full.npy`** en el **mismo MP4**.

---

## 5. Checklist de respuesta rápida (opcional)

Marcar Sí/No donde aplique:

| # | Afirmación | Sí / No |
|---|------------|---------|
| A | `user_*/poses_full` offline = equivalente a `merged.npy` C1 | **Sí** (objetivo; validar numéricamente; merged mask debe coincidir) |
| B | T = número de frames del `clip.mp4` @ 12 fps | **No** siempre — T = frames **decodificados** (fps del MP4); 12 fps solo si el MP4 es 12 CFR |
| C | Huecos = NaN en `poses_full`, no filas omitidas | **Sí** con `keep_all`; **No** con `frame_policy: iamodel` |
| D | Entrenamiento usa ventana 96 @ 12 Hz | **No** — usa **`seq_len` del modelo** (p. ej. 96/64) tras **`valid_mask` filter** sobre **clip entero** |
| E | `poses.npy` = `poses_full[valid_mask]` en disco | **Sí** |
| F | Re-extract de `clip_buffer` MinIO = gold para validar C1 | **Sí** |

**Comentarios:**

- **Chunks C1 ≠ unidades de entrenamiento.** Entrenamiento = **1 clip → 1 `user_*/poses_full`**. Prod motion = **N chunks/sesión** + inferencia **`clip` corta** → FP y auditoría confusa.
- Piloto oct-2026: chunks **&lt; 8 s**, **`pose_resampled`** sin llegar a 96, **`merged_valid_mask` ≠ chunk `valid_mask`** en cola — arreglar antes de usar merged para train.
- Paridad Triton OK; paridad **timeline + máscaras + unidad de scoring** pendiente.

---

## 6. Historial de pegado (iteraciones)

### Iteración 1 — fecha: 2026-10-19

Respuestas integradas en §1–§2 y checklist §5 (este documento). Detalle ampliado: `docs/C1_CONTINUOUS_EXPORT_TRAIN_PARITY_HANDOFF.md`.

### Iteración 2 — fecha: ______

```
(pegar aquí)
```

---

## 7. Referencias en repo C1

- [`POSE_EXPORT_IAMODEL.md`](./POSE_EXPORT_IAMODEL.md) — perfil MinIO `iamodel_full`
- [`MENSAJE_IAMODEL_TRITON_CLIP_EXTRACTOR_RESPUESTA.md`](./MENSAJE_IAMODEL_TRITON_CLIP_EXTRACTOR_RESPUESTA.md) — ClipExtractor vs C1 continuo
- [`C1_CONTINUOUS_EXPORT_TRAIN_PARITY_HANDOFF.md`](./C1_CONTINUOUS_EXPORT_TRAIN_PARITY_HANDOFF.md) — handoff chunks / merged
