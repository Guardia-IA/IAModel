#!/usr/bin/env bash
# Descarga pesos YOLO pose (.pt) en un directorio del host.
#
#   ./download_yolo_pose_pt.sh /home/angel/kanvis/piloto_v1/yolo
#   ./download_yolo_pose_pt.sh /home/angel/kanvis/piloto_v1/yolo yolo26s-pose.pt
#
# Orden: 1) Python del host (venv con ultralytics)  2) contenedor python:3.12-slim
# NO usa la imagen Triton (cv2/libxcb y pip rompen el entorno).

set -euo pipefail
DIR="${1:?usage: download_yolo_pose_pt.sh <host_dir> [model.pt]}"
NAME="${2:-yolo26s-pose.pt}"
DIR="$(cd "$DIR" && pwd)"
PYTHON_IMAGE="${PYTHON_IMAGE:-python:3.12-slim}"

if [[ -f "$DIR/$NAME" ]]; then
  echo "Ya existe: $DIR/$NAME ($(stat -c%s "$DIR/$NAME") bytes)"
  exit 0
fi

download_with_python() {
  local py="$1"
  "$py" - <<PY
import os, shutil
from pathlib import Path
dest = Path("$DIR") / "$NAME"
os.chdir("$DIR")
from ultralytics import YOLO
YOLO("$NAME")
if not dest.is_file():
    pts = sorted(Path("$DIR").rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pts:
        raise SystemExit("No apareció ningún .pt tras la descarga.")
    shutil.copy2(pts[0], dest)
print("OK", dest, dest.stat().st_size)
PY
}

echo "Descargando $NAME → $DIR"

if command -v python3 >/dev/null 2>&1; then
  if python3 -c "import ultralytics" 2>/dev/null; then
    echo "Usando python3 del host (ultralytics instalado)..."
    if download_with_python python3; then
      ls -la "$DIR/$NAME"
      exit 0
    fi
  fi
fi

echo "Host sin ultralytics usable → Docker $PYTHON_IMAGE ..."
docker run --rm \
  -v "$DIR:/work" \
  "$PYTHON_IMAGE" \
  bash -lc "
    set -e
    apt-get update -qq && apt-get install -y -qq libglib2.0-0 libgomp1 >/dev/null
    pip install -q 'numpy<2' ultralytics opencv-python-headless
    python3 - <<'PY'
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
        raise SystemExit('Descarga fallida')
    shutil.copy2(pts[0], dest)
print('OK', dest, dest.stat().st_size)
PY
  "

ls -la "$DIR/$NAME"
