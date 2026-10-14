#!/usr/bin/env bash
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

TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:25.02-py3}"
HOST_DIR=""
YOLO_PT="yolo26s-pose.pt"
OUT_NAME="yolo26s-pose.triton.plan"
IMGSZ=640
FP16="--fp16"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host-dir) HOST_DIR="$2"; shift 2 ;;
    --yolo) YOLO_PT="$2"; shift 2 ;;
    --output-name) OUT_NAME="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --no-fp16) FP16=""; shift ;;
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
  echo "ERROR: no existe $HOST_DIR/$YOLO_PT"
  exit 1
fi

echo "=== 1/2 Export ONNX (Ultralytics en contenedor Triton image + pip ultralytics) ==="
docker run --rm --gpus all \
  -v "$HOST_DIR:$WORK" \
  "$TRITON_IMAGE" \
  bash -lc "
    pip install -q ultralytics onnx 2>/dev/null || pip install ultralytics onnx
    python3 - <<'PY'
from ultralytics import YOLO
from pathlib import Path
pt = Path('$PT_PATH')
onnx = Path('$ONNX_PATH')
m = YOLO(str(pt))
m.export(format='onnx', imgsz=$IMGSZ, simplify=True, opset=17)
# Ultralytics escribe junto al .pt
generated = pt.with_suffix('.onnx')
if generated != onnx and generated.exists():
    generated.replace(onnx)
print('ONNX:', onnx, 'exists=', onnx.exists())
PY
  "

echo "=== 2/2 trtexec → plan TensorRT (misma imagen Triton) ==="
docker run --rm --gpus all \
  -v "$HOST_DIR:$WORK" \
  "$TRITON_IMAGE" \
  trtexec \
    --onnx="$ONNX_PATH" \
    --saveEngine="$PLAN_PATH" \
    $FP16 \
    --workspace=4096

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
