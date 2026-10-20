# Gold N2 — paridad decode MP4 (IAModel ↔ C1)

**Contrato piloto:** el offline gold debe decodificar **`clip_buffer.mp4`** como C1 **`clip_buffer_reinfer`** (`export_iamodel_poses_from_clip` en kanvis-monitoring): **OpenCV `VideoCapture`**.  
El **NVDEC GStreamer** del RTSP en vivo es **otro camino** (multi-cámara); no sustituye al decode del fichero en reinfer.

Contrato espejo en C1: `kanvis-monitoring/docs/IAMODEL_GOLD_N2_DECODE_PARITY.md`.

## Stack C1 (clip denso al cierre)

| Paso | C1 N2 |
|------|--------|
| Entrada | `clip_buffer.mp4` |
| Decode fichero | **OpenCV** `VideoCapture` |
| Infer | Triton, perfil reinfer, persona dominante |
| Normalización KP | espacio post-`infer_frame_max_*` (640×360) |
| Umbrales export IAModel full | 0,5 / 5 KP |
| Salida MinIO | `merged.npy`, `chunk_*` (`apply_clip_buffer_dense_pose_export`) |

## Cambios IAModel (`engine=c1_like`)

### 1. `experiments/triton_pose_extract/triton_pose_extract_config.json`

| Clave | Valor gold |
|-------|------------|
| `extract.decode_backend` | **`"opencv"`** |
| `extract.decode_c1_pixel_parity` | **`true`** |
| Comentario `_decode_backend_comment` | gold N2 = C1 reinfer; **`ffmpeg_nvdec`** opcional batch 5090 |

### 2. `triton_pose_extract/video_decode.py`

- `read_frames_bgr(..., decode_c1_pixel_parity=True)`: reconcilia **T** con **`ffprobe nb_read_frames`**; prueba varios backends si hace falta.
- **`_pick_decode_attempt(..., prefer_opencv=decode_c1_pixel_parity)`**:
  - Si `prefer_opencv`: orden **`opencv` → `ffmpeg_software` → `ffmpeg_nvdec`** entre intentos con T = ffprobe.
  - Si no: prioridad **nvdec** (batch masivo).

`c1_like_extract` pasa `decode_c1_pixel_parity` desde `pipe_cfg`.

### 3. `triton_pose_extract/pipeline_config.py` — `C1LikeExtractConfig`

- Default **`decode_backend`: `"opencv"`**.
- **`from_raw`**: override **`TRITON_EXTRACT_DECODE_BACKEND`** (`opencv` | `ffmpeg_nvdec`) si está definido; si no, JSON; si no, `opencv`.

### 4. Sin cambios en Triton / preprocess / 0,5+5 KP / `keep_all`

Solo decode + selección de backend.

## Uso

| Escenario | Decode |
|-----------|--------|
| **Gold N2 / compare MinIO** | `opencv` (config o env) |
| **Batch entrenamiento 5090** | `TRITON_EXTRACT_DECODE_BACKEND=ffmpeg_nvdec` o editar JSON |

Tras cambiar de nvdec → opencv hay que **re-extraer** offline; no basta repetir `compare_c1_golden` sobre npy viejos.

## Verificación (wudebian)

```bash
cd IAModel/experiments/triton_pose_extract
source ../training/venv/bin/activate
export TRITON_EXTRACT_DECODE_BACKEND=opencv   # opcional si JSON ya opencv

SESSION=/path/TestKanvis/SESSION_ID
OUT_BASE=/path/gold_offline_SESSION
OFFLINE="$OUT_BASE/gold/clip_buffer"

./run_extract.sh extract \
  --video "$SESSION/clip_buffer.mp4" \
  --out "$OUT_BASE" --cat gold --parallel 1 --validate

grep decode_backend_used "$OFFLINE/meta.json"
# → "opencv"

python3 compare_c1_golden.py \
  --c1-session "$SESSION" \
  --offline-clip "$OFFLINE" \
  --chunk chunk_001 --merged-align auto --align dense_index \
  --json "$SESSION/gold_report_n2.json"
```

**PASS merged N2:** `merged_whole_clip`: `align_mode` **`dense_index`**, **`pass_mask`** + **`pass_coords`**, ε = **1e-4**; informe con **`offline_decode_backend`: `opencv`**.

En C1 también existe `scripts/run_gold_n2_validation.sh` (exporta el mismo env y orquesta extract+compare).

## Referencias

- [`C1_NIVEL2_REINFER_MENSAJE.md`](./C1_NIVEL2_REINFER_MENSAJE.md)
- [`MENSAJE_IAMODEL_PARIDAD_POSES_FULL_C1.md`](./MENSAJE_IAMODEL_PARIDAD_POSES_FULL_C1.md) §10
