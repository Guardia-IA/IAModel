#!/usr/bin/env bash
# Wrapper IAModel → extracción Triton (ver README.md)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PY="${PYTHON:-python3}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

CMD="${1:-help}"
shift || true

case "$CMD" in
  check-triton)
    "$PY" extract_clips.py check-triton "$@"
    ;;
  preflight-csv)
    "$PY" extract_clips.py preflight-csv "$@"
    ;;
  extract-csv)
    "$PY" extract_clips.py extract-csv "$@"
    ;;
  bench-csv)
    "$PY" extract_clips.py bench-csv "$@"
    ;;
  bench-analyze)
    "$PY" extract_clips.py bench-analyze "$@"
    ;;
  extract)
    "$PY" extract_clips.py extract "$@"
    ;;
  smoke-one)
    "$PY" smoke_one_clip.py "$@"
    ;;
  compare-legacy)
    "$PY" scripts/compare_legacy_pose_npy.py "$@"
    ;;
  help|*)
    cat <<EOF
Uso: $0 <comando> [args]

  check-triton [--smoke]              Triton READY + capacidad paralela
  preflight-csv --input-root DIR      CSVs, clips/cat, tiempo estimado Triton
  extract-csv --input-root DIR        Batch CSV → FFmpeg → Triton (paralelo)
  bench-csv --input-root DIR          Muestra (50 clips) + ETA batch real
  bench-analyze REPORT.json           Re-proyectar ETA desde un informe
  extract --video FILE --out DIR      Un clip o carpeta suelta
  compare-legacy --video MP4 --reference .../poses_full.npy
                                      Re-extract batch + comparar npy antiguo
  smoke-one [--video PATH]            Prueba 1 clip

Variables:
  KANVIS_TRITON_POSE_ROOT  Ruta al repo kanvis-triton-pose en el HOST (pose_core Python)
  KANVIS_TRITON_URL        gRPC Triton (contenedor suele ser 127.0.0.1:47811 en el host)
  TRITON_EXTRACT_DECODE_BACKEND  opencv | ffmpeg_nvdec (paridad clip.mp4 → opencv)

Flujo típico (mismo equipo que Triton):
  $0 check-triton
  $0 preflight-csv \\
      --input-root /home/angel/DatosEntrenamiento/Videos/DatosOnline_3/sessions \\
      --output-base /media/8TB/.../data_triton_pose
  $0 extract-csv \\
      --input-root /ruta/carpeta1 --input-root /ruta/carpeta2 \\
      --output-base /media/8TB/.../data_triton_pose

Config: triton_pose_extract_config.json (filtros on/off, límites por categoría)

EOF
    ;;
esac
