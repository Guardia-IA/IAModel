#!/usr/bin/env bash
# Descarga pesos YOLO pose (.pt) en un directorio del host (Ultralytics + Docker).
#
#   ./download_yolo_pose_pt.sh /home/angel/kanvis/piloto_v1/yolo
#   ./download_yolo_pose_pt.sh /home/angel/kanvis/piloto_v1/yolo yolo26s-pose.pt

set -euo pipefail
DIR="${1:?usage: download_yolo_pose_pt.sh <host_dir> [model.pt]}"
NAME="${2:-yolo26s-pose.pt}"
DIR="$(cd "$DIR" && pwd)"
TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:25.02-py3}"

if [[ -f "$DIR/$NAME" ]]; then
  echo "Ya existe: $DIR/$NAME"
  exit 0
fi

echo "Descargando $NAME en $DIR ..."
docker run --rm \
  -v "$DIR:/work" \
  "$TRITON_IMAGE" \
  bash -lc "
    pip install -q ultralytics
    python3 - <<PY
import os, shutil
from pathlib import Path
os.chdir('/work')
name = '$NAME'
from ultralytics import YOLO
YOLO(name)
dest = Path('/work') / name
if not dest.is_file():
    pts = sorted(Path('/work').rglob('*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pts:
        raise SystemExit('Descarga fallida: no hay .pt en /work')
    shutil.copy2(pts[0], dest)
print('OK', dest, dest.stat().st_size)
PY
  "
ls -la "$DIR/$NAME"
