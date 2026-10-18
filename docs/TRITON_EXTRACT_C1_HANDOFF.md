# Handoff C1 / kanvis-triton-pose — extracción offline ClipExtractor (IAModel)

**Fuente de verdad código:** repo **`kanvis-triton-pose`** (`pose_core/clip_extractor.py`, `pose_core/tracker.py`, `config/pose.yaml`).  
**Producción continua (MinIO):** **`kanvis-monitoring`** — tubo distinto (colas, submuestreo); no sustituye extract CSV offline.

## Resumen ejecutivo

| Síntoma en logs | Causa | Acción IAModel |
|-----------------|-------|----------------|
| `skip clip …: single_user_only y 2 tracks` | Comportamiento **esperado** | `single_user_only: false` para multi-persona, o aceptar skip en batch single-user |
| `0 tracks (revisa Triton decode o tracker…)` | Decode/tracker sin `track_id` | Smoke Triton + golden clip; preprocess 640×360, `parse_conf`, warmup |
| Tracks pero sin `user_*` (sin log single_user) | Filtros `BODY_VISIBLE_*` | Mejorar clip o relajar filtros solo en lab |

Con **`single_user_only=true`**, más de un `track_id` → **no** exporta ningún `user_*` (no hay track dominante). Sí escribe `clip.mp4`, `meta.json`, `pose_bundle.json`; `users: []`.

## Tracker `.pt` + Triton

- **Pose:** Triton gRPC `yolo26s-pose` (TRT).
- **ByteTrack:** `yolo26s-pose.pt` local solo como contenedor Ultralytics del tracker; keypoints vienen de Triton.

## `meta.json` sin `skip_reason` (v1)

Diagnosticar con log `-v` + `users` vacío vs poblado. Ver ejemplos A/B/C en respuesta C1 original (Oct 2026).

## Checklist batch (piloto)

1. `curl …/v2/models/yolo26s-pose/ready` → 200  
2. `check_triton_pose_batch.py --url 127.0.0.1:47811`  
3. Golden clip vendor / un clip referencia con `--no-single-user-only` primero  
4. Política job: skip single-user ≠ fallo infra  

## IAModel

- Config: `experiments/triton_pose_extract/triton_pose_extract_config.json` → `filters.single_user_only`  
- CLI override: `extract-csv --no-single-user-only`  
- Informe: `logs/csv_extract_report.json` → `jobs_ok`, `jobs_skip_no_export`, `jobs_fail_infra`  

Referencias: [AGENT_TRITON_NPY_BRIEF.md](./AGENT_TRITON_NPY_BRIEF.md), [TRITON_EXTRACT_PERF.md](./TRITON_EXTRACT_PERF.md), README `experiments/triton_pose_extract/`.
