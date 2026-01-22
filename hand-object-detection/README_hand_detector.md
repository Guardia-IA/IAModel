# Detector de Interacción Mano-Objeto

Solución moderna para detectar cuando un usuario coge un objeto en videos de supermercado, optimizada para distancias de ~5 metros.

## Características

- **MediaPipe**: Detección precisa de manos incluso a distancia
- **YOLO (Ultralytics)**: Detección de objetos genéricos sin necesidad de clasificación
- **Tracking temporal**: Reduce falsos positivos confirmando interacciones en múltiples frames
- **Visualización en tiempo real**: Muestra resultados con OpenCV
- **Optimizado para distancias largas**: Funciona bien con cámaras de supermercado

## Instalación

```bash
pip install -r requirements.txt
```

## Uso Básico

### Desde línea de comandos:

```bash
python hand_object_detector.py ruta/a/tu/video.mp4
```

### Desde código Python:

```python
from hand_object_detector import HandObjectDetector

# Crear detector
detector = HandObjectDetector(
    video_path='ruta/a/tu/video.mp4',
    yolo_model='yolov8n.pt',  # Modelo ligero
    interaction_threshold=80,  # Distancia en píxeles (ajustar según resolución)
    min_frames_interaction=3,  # Frames consecutivos para confirmar
    confidence_hands=0.5,      # Confianza para manos
    confidence_objects=0.3     # Confianza para objetos
)

# Procesar video
detector.process_video(show_video=True)
```

## Parámetros Ajustables

- **`yolo_model`**: Modelo YOLO a usar
  - `yolov8n.pt` (nano) - Más rápido, menos preciso
  - `yolov8m.pt` (medium) - Balance velocidad/precisión
  - `yolov8l.pt` (large) - Más preciso, más lento
  - `yolov10n.pt`, `yolov10m.pt` - Versiones más recientes

- **`interaction_threshold`**: Distancia en píxeles para considerar interacción
  - Para videos 720p a 5m: ~80-100 píxeles
  - Para videos 1080p a 5m: ~120-150 píxeles
  - Ajustar según resolución y distancia real

- **`min_frames_interaction`**: Mínimo de frames consecutivos para confirmar
  - Mayor valor = menos falsos positivos pero más lento en detectar
  - Recomendado: 3-5 frames

- **`confidence_hands`**: Confianza mínima para detección de manos (0.0-1.0)
  - Para distancias largas: reducir a 0.4-0.5

- **`confidence_objects`**: Confianza mínima para detección de objetos (0.0-1.0)
  - Para objetos pequeños a distancia: reducir a 0.25-0.3

## Opciones de Línea de Comandos

```bash
python hand_object_detector.py video.mp4 \
    --model yolov8m.pt \
    --threshold 100 \
    --min-frames 5 \
    --conf-hands 0.4 \
    --conf-objects 0.25 \
    --output video_procesado.mp4 \
    --no-show
```

## Visualización

El código muestra:
- **Manos**: Puntos rojos (landmarks) y círculo amarillo (centro)
- **Objetos**: Rectángulos verdes con confianza
- **Interacciones**: Líneas conectando mano y objeto
  - Amarillo: Interacción detectada (pendiente de confirmar)
  - Rojo: Interacción confirmada (objeto cogido)
- **Texto**: "OBJETO COGIDO DETECTADO" cuando se confirma

## Controles durante la visualización

- **'q'**: Salir
- **'p'**: Pausar/reanudar

## Integración con tu Sistema

Este detector complementa tu modelo de pose tracking:

1. **Tu modelo de pose**: Detecta actividades generales (compra, coger, guardar)
2. **Este detector**: Confirma interacción mano-objeto específica
3. **Combinación**: Reduce falsos positivos significativamente

Ejemplo de integración:

```python
# Tu código de pose tracking
pose_result = tu_modelo_pose.detect(frame)

# Detección mano-objeto
detector = HandObjectDetector(video_path)
frame, interactions, confirmed = detector.process_frame(frame)

# Combinar resultados
if pose_result == "coger_objeto" and confirmed:
    # Alta confianza de robo
    alerta_robo()
```

## Notas sobre Distancias Largas

Para videos a 5 metros de distancia:

1. **Ajusta `interaction_threshold`** según la resolución del video
2. **Reduce `confidence_objects`** a 0.25-0.3 para detectar objetos pequeños
3. **Usa modelos YOLO más grandes** (`yolov8m.pt` o `yolov8l.pt`) si tienes GPU
4. **Aumenta `min_frames_interaction`** a 5-7 para mayor robustez

## Solución vs. Alternativas Antiguas

Esta solución es superior a `hand_object_detector` (2018) porque:
- ✅ Usa modelos actualizados (2023-2024)
- ✅ MediaPipe optimizado para distancias variables
- ✅ YOLO detecta objetos genéricos sin entrenamiento
- ✅ Tracking temporal reduce falsos positivos
- ✅ Activamente mantenido y actualizado

## Troubleshooting

**Las manos no se detectan:**
- Reduce `confidence_hands` a 0.4
- Verifica que las manos sean visibles en el video
- A 5m puede ser necesario aumentar resolución

**Los objetos no se detectan:**
- Reduce `confidence_objects` a 0.25
- Usa un modelo YOLO más grande (`yolov8m.pt`)
- Verifica que los objetos sean visibles

**Muchos falsos positivos:**
- Aumenta `min_frames_interaction` a 5-7
- Aumenta `interaction_threshold` (más estricto)
- Aumenta `confidence_objects`

**Procesamiento lento:**
- Usa `yolov8n.pt` (modelo nano)
- Reduce resolución del video antes de procesar
- Usa GPU si está disponible
