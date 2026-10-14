# Engine - Generación de .engine TensorRT para YOLO

Genera ficheros `.engine` optimizados para tu GPU NVIDIA. La inferencia con .engine suele ser **2-5x más rápida** que con .pt.

## Requisitos

- **GPU NVIDIA** con CUDA
- Driver NVIDIA actualizado
- CUDA Toolkit (11.x o 12.x)
- TensorRT (apt o pip)

## Instalación

```bash
# 1. PyTorch con CUDA (según tu versión)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Dependencias del paquete
pip install -r engine/requirements.txt

# 3. TensorRT (Ubuntu)
sudo apt install nvidia-tensorrt
# o pip install nvidia-tensorrt (si hay wheel para tu plataforma)
```

## Uso

### Un solo engine en ruta del host (C1 / contenedores compartidos)

```bash
cd IAModel/experiments
python engine/export_yolo_pose_engine.py \
  --output-dir /opt/kanvis/models/yolo26s-pose/v1 \
  --yolo yolo26s-pose
```

Genera `yolo26s-pose.engine` y `yolo26s-pose.engine.manifest.json` (hash, GPU, imgsz, versiones).

**Importante — C1 Triton pose:** el `.engine` de `export_yolo_pose_engine.py` / Ultralytics **no** es el mismo formato que el `model.plan` de Triton (`tensorrt_plan`). Triton espera un plan compilado con **trtexec en la imagen** `nvcr.io/nvidia/tritonserver:25.02-py3` (u la que use el piloto). Copiar el `.engine` Ultralytics a `model.plan` provoca fallo de deserialización (magic 833 vs plan TRT).

Para piloto wudalkot use `build_triton_pose_plan.sh`:

```bash
chmod +x engine/build_triton_pose_plan.sh
./engine/build_triton_pose_plan.sh \
  --host-dir /home/angel/kanvis/piloto_v1/yolo \
  --yolo yolo26s-pose.pt \
  --output-name yolo26s-pose.triton.plan
```

Montar **ese** `.triton.plan` (o renombrarlo según contrato C1 tras validación de formato). La extracción offline IAModel puede seguir usando `.pt` o un `.engine` Ultralytics local; no mezclar ficheros.

### Varios modelos (carpeta `engine/` local)

```bash
# Generar engines por defecto (n y m)
python engine/build_engines.py

# Modelo concreto
python engine/build_engines.py --model yolo11x-pose.pt

# Varios modelos
python engine/build_engines.py --model yolo11n-pose.pt yolo11m-pose.pt yolo11x-pose.pt

# FP32 (sin half)
python engine/build_engines.py --model yolo11m-pose.pt --no-half
```

Los `.engine` se guardan en la propia carpeta `engine/`. `pose_extractor_clean.py` los busca automáticamente: si existe `engine/yolo11m-pose.engine`, lo usa; si no, usa el `.pt`.

**Importante:** Los .engine son específicos de la GPU. Si cambias de máquina o de versión de CUDA, hay que regenerarlos.
