#!/usr/bin/env bash
# AVISO (15-oct-2026): para el piloto, la vía OFICIAL es la de C1 (kanvis-monitoring):
#   scripts/export_triton_pose_onnx.py (dynamic=True, batch=32, end2end, opset 17)
#   + scripts/triton_pose_entrypoint.sh (trtexec con shapes 1/16/32 e inputIOFormats fp16:chw).
# Este script exporta ONNX estático y compila sin perfiles de batch: su plan NO es compatible
# con el config.pbtxt de C1 (max_batch 32, entrada FP16). Úsalo solo para pruebas locales.
#
# Genera un model.plan (TensorRT crudo) compatible con Triton nvcr.io/nvidia/tritonserver:25.02-py3
# a partir de yolo26s-pose.pt (u otro YOLO pose).
#
# NO uses el .engine de Ultralytics (export format=engine) como model.plan: lleva cabecera
# propia (magic 833 / JSON) y Triton tensorrt_plan no lo deserializa.
#
# Uso (en máquina con Docker + GPU NVIDIA, misma familia que piloto wudalkot):
#
#   ./build_triton_pose_plan.sh \
#     --host-dir /home/angel/kanvis/piloto_v1/yolo \
#     --yolo yolo26s-pose.pt \
#     --output-name yolo26s-pose.triton.plan
#
# Luego en C1: apuntar YOLO_POSE_ENGINE (o la var que mapee a model.plan) a ese .triton.plan
# (o copiarlo como yolo26s-pose.engine SOLO si C1 valida formato TRT plan, no Ultralytics).
#
# Requisitos: nvidia-container-toolkit, acceso a GPU en docker run --gpus all

set -euo pipefail

# Imagen Triton server NO incluye trtexec en PATH; compilar plan con TensorRT 25.02 (misma major que sidecar).
TRTEXEC_IMAGE="${TRTEXEC_IMAGE:-nvcr.io/nvidia/tensorrt:25.02-py3}"
PYTHON_IMAGE="${PYTHON_IMAGE:-python:3.12-slim}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOST_DIR=""
YOLO_PT="yolo26s-pose.pt"
OUT_NAME="yolo26s-pose.triton.plan"
IMGSZ=640
FP16="--fp16"
AUTO_DOWNLOAD=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host-dir) HOST_DIR="$2"; shift 2 ;;
    --yolo) YOLO_PT="$2"; shift 2 ;;
    --output-name) OUT_NAME="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --no-fp16) FP16=""; shift ;;
    --no-download) AUTO_DOWNLOAD=0; shift ;;
    *) echo "Arg desconocido: $1"; exit 1 ;;
  esac
done

if [[ -z "$HOST_DIR" ]]; then
  echo "ERROR: --host-dir obligatorio (directorio en el host, p. ej. /home/angel/kanvis/piloto_v1/yolo)"
  exit 1
fi

HOST_DIR="$(cd "$HOST_DIR" && pwd)"
WORK="/work"
PT_PATH="$WORK/$YOLO_PT"
ONNX_PATH="$WORK/${YOLO_PT%.pt}.onnx"
PLAN_PATH="$WORK/$OUT_NAME"

if [[ ! -f "$HOST_DIR/$YOLO_PT" ]]; then
  if [[ "$AUTO_DOWNLOAD" -eq 1 ]]; then
    "$SCRIPT_DIR/download_yolo_pose_pt.sh" "$HOST_DIR" "$YOLO_PT"
  else
    echo "ERROR: no existe $HOST_DIR/$YOLO_PT (usa --no-download solo si ya tienes el .pt)"
    exit 1
  fi
fi
if [[ ! -f "$HOST_DIR/$YOLO_PT" ]]; then
  echo "ERROR: sigue sin existir $HOST_DIR/$YOLO_PT tras descarga"
  exit 1
fi

echo "=== 1/2 Export ONNX (python slim / host; no imagen Triton — evita libxcb/cv2) ==="
if command -v python3 >/dev/null 2>&1 && python3 -c "import ultralytics" 2>/dev/null; then
  python3 - <<PY
from ultralytics import YOLO
from pathlib import Path
pt = Path("$HOST_DIR/$YOLO_PT")
onnx = Path("$HOST_DIR/${YOLO_PT%.pt}.onnx")
m = YOLO(str(pt))
m.export(format="onnx", imgsz=$IMGSZ, simplify=True, opset=17)
gen = pt.with_suffix(".onnx")
if gen != onnx and gen.exists():
    gen.replace(onnx)
print("ONNX:", onnx, onnx.exists())
PY
else
  docker run --rm \
    -v "$HOST_DIR:$WORK" \
    "$PYTHON_IMAGE" \
    bash -lc "
      apt-get update -qq && apt-get install -y -qq libglib2.0-0 libgomp1 >/dev/null
      pip install -q 'numpy<2' ultralytics onnx opencv-python-headless
      python3 - <<'PY'
from ultralytics import YOLO
from pathlib import Path
pt = Path('$PT_PATH')
onnx = Path('$ONNX_PATH')
m = YOLO(str(pt))
m.export(format='onnx', imgsz=$IMGSZ, simplify=True, opset=17)
gen = pt.with_suffix('.onnx')
if gen != onnx and gen.exists():
    gen.replace(onnx)
print('ONNX:', onnx, onnx.exists())
PY
    "
fi

echo "=== 2/2 trtexec → plan TensorRT ($TRTEXEC_IMAGE; --gpus all + driver NVIDIA) ==="
docker run --rm --gpus all \
  -v "$HOST_DIR:$WORK" \
  --entrypoint bash \
  "$TRTEXEC_IMAGE" \
  -lc "
    TRTEXEC=\$(command -v trtexec || true)
    [[ -z \"\$TRTEXEC\" && -x /usr/src/tensorrt/bin/trtexec ]] && TRTEXEC=/usr/src/tensorrt/bin/trtexec
    [[ -z \"\$TRTEXEC\" ]] && { echo 'ERROR: trtexec no encontrado en imagen'; exit 1; }
    \"\$TRTEXEC\" \
      --onnx='$ONNX_PATH' \
      --saveEngine='$PLAN_PATH' \
      $FP16 \
      --workspace=4096
  "

echo ""
echo "Plan generado: $HOST_DIR/$OUT_NAME"
echo "Verificar magic (debe ser plan TRT, no 833 Ultralytics):"
python3 - <<PY
import struct, sys
p = "$HOST_DIR/$OUT_NAME"
with open(p, "rb") as f:
    b = f.read(4)
le = struct.unpack("<I", b)[0]
print(f"  first u32 LE = {le} (1953657958 = 0x7472746e 'trt\\n' típico plan; 833 = formato Ultralytics .engine)")
PY
echo ""
echo "Siguiente: configurar C1 para montar ESTE fichero como model.plan (sync Triton pose)."
echo "Mantén un .engine Ultralytics aparte solo si lo usáis en extracción offline IAModel (no en Triton)."
