# Integración del Detector de ddshan

Este proyecto ahora incluye soporte para el detector de [ddshan/hand_object_detector](https://github.com/ddshan/hand_object_detector), un detector basado en Faster R-CNN específicamente entrenado para detectar manos y objetos en contacto.

## Ventajas del Detector de ddshan

- **Especializado**: Entrenado específicamente para detectar manos y objetos en contacto
- **Alta precisión**: 90.4% AP para manos, 66.3% AP para objetos en el dataset 100K+ego
- **Información adicional**: Detecta estado de contacto (N/S/O/P/F) y lado de la mano (L/R)

## Requisitos para usar ddshan

1. **Compilar código CUDA**:
```bash
cd ddshan_detector/lib
python setup.py build develop
```

2. **Descargar modelo pre-entrenado**:
   - Modelo recomendado: `handobj_100K+ego` (mejor para datos egocéntricos)
   - Link: https://drive.google.com/open?id=1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE
   - Guardar en: `ddshan_detector/models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth`

3. **Instalar dependencias adicionales**:
```bash
pip install easydict cython cffi msgpack matplotlib pyyaml tensorboardX tqdm jinja2 pascal_voc_writer
```

## Uso

### Desde línea de comandos:

```bash
# Usar detector de ddshan
python hand_object_detector.py /ruta/a/video.mp4 --use-ddshan

# Especificar ruta al modelo manualmente
python hand_object_detector.py /ruta/a/video.mp4 --use-ddshan --ddshan-model /ruta/al/modelo.pth
```

### Desde código Python:

```python
from hand_object_detector import HandObjectDetector

# Usar detector de ddshan
detector = HandObjectDetector(
    video_path='video.mp4',
    use_ddshan=True,
    ddshan_model_path='ddshan_detector/models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth',
    confidence_hands=0.5,  # Umbral para manos
    confidence_objects=0.5  # Umbral para objetos
)

detector.process_video(show_video=True)
```

## Comparación de Métodos

### Método Original (MediaPipe + YOLO)
- ✅ No requiere compilación CUDA
- ✅ Más rápido
- ✅ Funciona sin modelos adicionales
- ❌ Menos preciso para detección de contacto mano-objeto

### Método ddshan (Faster R-CNN)
- ✅ Especializado en contacto mano-objeto
- ✅ Mayor precisión
- ✅ Información adicional (estado de contacto, lado de mano)
- ❌ Requiere compilación CUDA
- ❌ Requiere descargar modelo pre-entrenado
- ❌ Más lento

## Estados de Contacto (ddshan)

- **N**: No contact (sin contacto)
- **S**: Self contact (contacto con uno mismo)
- **O**: Other person contact (contacto con otra persona)
- **P**: Portable object contact (contacto con objeto portátil) ⭐ **Este es el que buscamos**
- **F**: Stationary object contact (contacto con objeto estacionario)

## Notas

- Si el detector de ddshan no está disponible (no compilado o modelo no encontrado), el sistema automáticamente usará el método original (MediaPipe + YOLO)
- El detector de ddshan funciona mejor con videos donde las manos y objetos son claramente visibles
- Para videos de supermercado a 5 metros, puede que necesites ajustar los umbrales de confianza
