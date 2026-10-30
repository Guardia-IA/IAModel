# Extracción pose IAModel vía Triton (`kanvis-triton-pose`)

Reemplazo del flujo **`pose_extractor_clean.py`** (YOLO local) por **Triton + `pose_core`**, con el mismo **CSV → recorte FFmpeg → npy** y **preflight** de conteos/tiempos.

Handoff: [`../../docs/AGENT_TRITON_NPY_BRIEF.md`](../../docs/AGENT_TRITON_NPY_BRIEF.md). Rendimiento / NVDEC / C1: [`../../docs/TRITON_EXTRACT_PERF.md`](../../docs/TRITON_EXTRACT_PERF.md).  
Gold C1 vs offline (**kanvis_pose_v1**): [`../../docs/MENSAJE_IAMODEL_PARIDAD_POSES_FULL_C1.md`](../../docs/MENSAJE_IAMODEL_PARIDAD_POSES_FULL_C1.md) §10.2; script `compare_c1_golden.py`.

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

## Motor `c1_like` (más rápido, solo IAModel)

Por defecto en `triton_pose_extract_config.json` → **`extract`**:

| Campo | Default | Efecto |
|--------|---------|--------|
| `engine` | `c1_like` | NVDEC + preprocess paralelo **sin tocar** el repo vendor |
| `decode_backend` | `opencv` (gold N2) | Gold = paridad C1 `clip_buffer_reinfer`; `ffmpeg_nvdec` batch 5090 |
| `preprocess_workers` | 8 | Letterbox como `triton_preprocess_workers` en C1 |
| `fallback_opencv_decode` | true | Si NVDEC falla, OpenCV |

Triton sigue igual (gRPC + mismas primitivas `pose_core`: infer, decode, ByteTrack, export).  
Volver al camino antiguo: `"engine": "clip_extractor"`.

En `meta.json` verás `iamodel_extract_engine: c1_like` en clips nuevos.

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

# Debug: primeros N clips (`--limit` usa máx. 2 workers; `--parallel 1` = log más claro)
./run_extract.sh extract-csv --input-root ... --output-base ... --limit 5 --parallel 1 -v

## Bench / ETA (más fiable que preflight)

Procesa **50 clips reales** (respeta `skip_existing`), mide mediana/p90 y proyecta horas restantes:

```bash
./run_extract.sh bench-csv \
  --input-root /ruta/csv ... \
  --output-base /media/8TB/.../data_triton_pose \
  --sample 50 --parallel 2 \
  --total-clips 22409 -v
```

Sin `--with-inventory` (default): **no** escanea 22k filas antes (evita ~7 min en 8TB).  
`triton.use_system_shm` debe ser **`false`** (pose_core actual no lo implementa).

Informe: `{output_base}/logs/bench_extract_YYYYMMDD_HHMMSS.json`

Pending **exacto** (resta ya exportados; **lento** — construye cola completa):

```bash
./run_extract.sh bench-csv ... --sample 50 --count-pending -v
```

Re-analizar un informe con pendientes manuales (p.ej. tras contar npy):

```bash
./run_extract.sh bench-analyze logs/bench_extract_....json --pending-jobs 15686
```

# Calentar Triton antes del batch (1ª infer GPU puede tardar 1–3 min)
./run_extract.sh check-triton --smoke
```

**¿Por qué tarda al inicio?** Cada worker hace `triton_connected` y luego `pose_core` carga **ByteTrack + `yolo26s-pose.pt` en disco** (no sustituye la inferencia TRT en Triton). Eso explica ~1 min de silencio con 4 workers.

**`[SKIP] sin carpetas user_*` / `single_user_only y 2 tracks`:** el clip se procesó pero **no se exportó npy** (varias personas en cámara o filtros de calidad). Es normal en `clip_buffer` de sesiones online; no es fallo de red Triton. Para prueba rápida usa un `.mp4` con una persona: `./run_extract.sh smoke-one --video …`

**Comparar datos antiguos vs extract Triton actual (mismo pipeline que `extract-csv`):**

| Qué | Dónde |
|-----|--------|
| **Triton** | Contenedor Docker `triton-pose` (inferencia GPU, puerto **47811** gRPC) |
| **Extract Python** | **Host** (wudebian): venv + `~/IAModel/experiments/triton_pose_extract` |
| **`KANVIS_TRITON_POSE_ROOT`** | Clone **`kanvis-triton-pose`** en disco del host (librería `pose_core`), **no** va dentro del contenedor Triton |

```bash
cd IAModel/experiments/triton_pose_extract
source ../training/venv/bin/activate
export KANVIS_TRITON_POSE_ROOT=~/kanvis/piloto_v1/kanvis-triton-pose   # ajustar
export KANVIS_TRITON_URL=127.0.0.1:47811
./run_extract.sh check-triton
./run_extract.sh compare-legacy \
  --video /ruta/data_result/CAT/clip/clip.mp4 \
  --reference /ruta/data_result/CAT/clip/user_0/poses_full.npy
```

Batch masivo (como preparaste clips): `./run_extract.sh extract-csv --input-root … --output-base …` — el compare usa el mismo motor **`c1_like`** + config JSON, **sin** `--session-close`.

**Cierre sesión C1 + sidecar:** kanvis-triton-pose con `pose_core.session_close_track_vote` (`run_session_close_track_vote_export`, contrato **`session_close_track_vote_v1`**: ByteTrack offline → votos IoU en anclas → export track ganador **0…T−1**). IAModel `c1_like` llama la misma API que `ClipExtractor`; `session_close_min_iou=0.3`. Ver `IAModel/docs/AGENT_TRITON_NPY_BRIEF.md` (§ `--session-close`).

**Cierre sesión C1 (clip = plano entero, 1 persona por sesión):** no uses `--no-single-user-only` (desactiva `export_primary_user`). Usa:

```bash
./run_extract.sh extract --video clip.mp4 --out … --cat gold --validate --session-close
# opcional, track ByteTrack de la sesión:
./run_extract.sh extract … --export-track-id 3 --validate
```

Log `N track(s) filtros (filtered=N)` = **BODY_VISIBLE_MIN_FRAMES/RATIO** de `pose_core`, no `single_user_only` (ese mensaje es `single_user_only y N tracks → sin export`).
```

Informes: `{output_base}/logs/csv_extract_report.json`, preflight JSON opcional.

**Montaje `/media/…` y `PermissionError` en `clip.mp4`:** algunos discos no permiten `utime` tras `shutil.copy2` en `pose_core`. IAModel parchea `copy2` → fallback a `copyfile` al cargar el vendor. Si un clip falló a medias, borra su carpeta en `data_result/…` y relanza.

## Un solo vídeo

```bash
./run_extract.sh extract --video /ruta/clip.mp4 --out ./out_test --validate
```
