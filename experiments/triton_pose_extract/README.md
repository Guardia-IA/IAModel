# Extracción pose IAModel vía Triton (`kanvis-triton-pose`)

Reemplazo del flujo **`pose_extractor_clean.py`** (YOLO local) por **Triton + `pose_core`**, con el mismo **CSV → recorte FFmpeg → npy** y **preflight** de conteos/tiempos.

Handoff: [`../../docs/AGENT_TRITON_NPY_BRIEF.md`](../../docs/AGENT_TRITON_NPY_BRIEF.md).

## Despliegue en el mismo host que Triton

Por defecto en `triton_pose_extract_config.json`:

- gRPC: **`kanvis-pilot.ai:47811`**
- HTTP ready: **`http://kanvis-pilot.ai:47810`**

## Requisitos

- Repo **`kanvis-triton-pose`** (`KANVIS_TRITON_POSE_ROOT` o clon en `../kanvis-triton-pose`).
- `pip install -r requirements.txt` (+ dependencias del vendor).
- **ffmpeg**, **pandas**, OpenCV.

## CSV (igual que antes)

Cada fila: `video_path, t_start, t_end, category`

- `00:00:00` + `00:00:00` → vídeo **completo** (sin recortar).
- Otros rangos → **FFmpeg** genera clip temporal → Triton.

Salida: `{output_base}/data_result/{cat}/{clip_name}/user_*/poses_full.npy`, `meta.json`, etc.

Nombres de carpeta **cortos** (fecha del path + stem + HMS + `c{cat}` + hash 6 chars), no la ruta absoluta del CSV.

## Filtros (activar/desactivar)

En `triton_pose_extract_config.json` → sección **`filters`**:

| Filtro | Default | Notas |
|--------|---------|--------|
| `single_user_only` | true | En `pose_core` (descarta multi-track) |
| `min_clip_duration_sec` | 0.5 s | enabled/value |
| `max_clip_duration_sec` | 120 s | clips muy largos |
| `min_user_seconds` | off | reservado (post) |
| `min_coverage_ratio` | 0.8 | alineado legacy; cuerpo visible en export |
| `skip_existing_output` | true | reanudación |
| `category_limits` | `../config_pose_extraction.json` | máx. clips por cat |

Cuerpo visible / kp conf: **`pose_core`** (`BODY_VISIBLE_MIN_*`, mismo criterio que producción).

## Paralelismo

Auto: `min(gpu_instance_count×2, max_batch_size // triton_batch_size)` desde `config/pose.yaml` del vendor (~**4 workers** en piloto).

Override: `--parallel N` o `"parallel_clips": N` en JSON.

Cada worker = **1 cliente gRPC + 1 tracker** por hilo.

## Comandos

```bash
cd IAModel/experiments/triton_pose_extract
chmod +x run_extract.sh

./run_extract.sh check-triton

# Preflight (conteo + tiempo)
./run_extract.sh preflight-csv \
  --input-root /media/8TB/DatosEntrenamiento/datos_3/csv_buenos \
  --output-base /media/8TB/DatosEntrenamiento/data_clips/data_triton_pose \
  --probe-videos \
  --save-json ./logs/preflight_triton.json

# Extracción batch (como pose_extractor_clean)
./run_extract.sh extract-csv \
  --input-root /media/8TB/DatosEntrenamiento/datos_3/csv_buenos \
  --output-base /media/8TB/DatosEntrenamiento/data_clips/data_triton_pose

# Debug: primeros N clips
./run_extract.sh extract-csv --input-root ... --output-base ... --limit 5
```

Informes: `{output_base}/logs/csv_extract_report.json`, preflight JSON opcional.

**Montaje `/media/…` y `PermissionError` en `clip.mp4`:** algunos discos no permiten `utime` tras `shutil.copy2` en `pose_core`. IAModel parchea `copy2` → fallback a `copyfile` al cargar el vendor. Si un clip falló a medias, borra su carpeta en `data_result/…` y relanza.

## Un solo vídeo

```bash
./run_extract.sh extract --video /ruta/clip.mp4 --out ./out_test --validate
```
