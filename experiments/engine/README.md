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
