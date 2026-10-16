# KANVIS Pose — proyecto compartido (motor de inferencia + núcleo de extracción de pose)

**Documento autocontenido para el agente que implementa el proyecto nuevo.**
Fecha: 15-oct-2026. Origen: análisis conjunto IAModel (entrenamiento) ↔ C1 (`kanvis-monitoring`, producción).

> **Cómo leer este documento.** §1–§3 explican el porqué. §4–§8 son la especificación (arquitectura, contratos, API).
> §9 son las fases con criterios de aceptación. §10–§12 son tests e integración. Los anexos contienen el código de
> referencia, los YAML y la lista de ficheros a traer de los otros repos. Lo marcado como **[VERIFICAR]** no está confirmado.
> Los números marcados como *propuesta* son valores iniciales a calibrar, no requisitos demostrados.

---

## 0. Resumen en 10 líneas

1. Kanvis detecta robos en supermercado a partir de secuencias de pose (8 keypoints) extraídas con **YOLO26s-pose**.
2. Hoy hay **dos extractores distintos**: uno offline (IAModel, Ultralytics) para generar datos de entrenamiento, y uno en producción (C1, Triton + TensorRT). Producen `.npy` que **no son equivalentes**.
3. Ese desajuste es una causa probable de falsos positivos en producción (el modelo se entrena con datos distintos de los que ve).
4. Se ha detectado un **posible error grave en C1**: envía imágenes en **BGR** al modelo, que espera **RGB** (ver §2).
5. Objetivo: **un único pipeline de pose versionado**, usado por producción (C1) y por la extracción offline de datos de entrenamiento.
6. Este proyecto nuevo contiene (a) la construcción reproducible del **modelo + imagen Triton** y (b) el paquete Python **`pose_core`** (preproceso, decode, tracker, export, bundle, CLI batch).
7. C1 e IAModel pasan a **consumir** este proyecto. El camino caliente de C1 (batching entre cámaras, SHM, colas) **no se mueve**.
8. Se entrega por fases, con tests de paridad (golden) y de deriva antes de tocar producción.
9. No se reentrena nada hasta medir la deriva entre extractores.
10. El `pose_bundle_id` identifica qué versión generó cada `.npy`.

---

## 1. Contexto

### 1.1 Producto y cascada
- Una tienda, nueve cámaras. Detección de robo por cascada: **Stage-1 motion** (pose → modelo temporal) → HAR → SAM3 → C3 (fusión).
- Stage-1 actual: ensemble de dos modelos (`model.pt`, `model2.pt`), regla AND con umbral 0.80, sobre `poses_full.npy`.
- Objetivo de negocio: ≤1 falso positivo por cada 1.000 sesiones sin robo y ≥66 % de recall de eventos.
  Estado reportado: ~20 FP en 654 sesiones en producción (≈3 %), frente a ~0,7 % en validación offline.

### 1.2 Piezas actuales (nombres que usa el equipo)
| Nombre | Qué es | Repo |
|---|---|---|
| **IAModel** | Entrenamiento y evaluación de los modelos de movimiento. Contiene `pose_extractor_clean.py`. | repo IAModel (este proyecto lo consume) |
| **C1** | Producción: RTSP → pose (Triton) → tracker → sesiones → chunks `.npy` en MinIO. | repo `kanvis-monitoring` |
| **C2** | Aplica el modelo de movimiento sobre los `.npy` de C1 (otro Triton). **No ejecuta YOLO.** | fuera de alcance |

### 1.3 Qué datos se usan para entrenar
- Cada muestra de entrenamiento es **un clip curado de una sola persona, ya recortado a una acción**, etiquetado con una categoría.
- De cada clip se obtiene **un `.npy`** (o uno por persona detectada) con `pose_extractor_clean.py`.
- Por tanto el extractor offline procesa **clips cortos y sueltos**, no streams continuos.

---

## 2. Hallazgos que motivan el proyecto

### 2.1 Tabla de diferencias entrenamiento ↔ producción
Valores tomados del código de ambos lados (ver Anexos A y C). Orden = prioridad estimada (**hipótesis a medir**).

| # | Factor | Entrenamiento (IAModel) | Producción (C1) |
|---|---|---|---|
| 1 | **Orden de canales** | RGB (preproceso estándar de Ultralytics) | **BGR sin convertir** hacia Triton (`chw_normalized` solo hace `transpose` y `/255`) |
| 2 | **Tracker** | BoT-SORT + ReID (`new_track_thresh 0.7`, `track_buffer 300`, `match_thresh 0.5`, GMC `sparseOptFlow`) | **ByteTrack** (`bytetrack_long.yaml`: `new_track 0.25`, `track_buffer 9000`, `match 0.92`, `low 0.05`) |
| 3 | **Frames sin ID de track** | **Descartados** | No se descartan; si no hay ID entero se asigna un ID por orden izquierda→derecha |
| 4 | **Eje temporal** | Todos los frames del vídeo | ~15 fps de inferencia en sesión (~4 fps en idle), con posible **hold-last** (`is_synthetic`) |
| 5 | **Confianza de detección** | `track()` de Ultralytics fuerza `conf=0.1` por defecto (verificado en 8.4.x) | decode ≥ `max(0.05, conf×0.5)`; luego filtro a 0.18 (sesión activa), 0.22 (idle) o 0.30 (por defecto) |
| 6 | **Inferencia** | Ultralytics local, `half=True`, `.engine` o `.pt` | Triton + plan TensorRT, entrada FP16, salida end2end `(B,300,57)` |
| 7 | **Resolución** | FFmpeg reescala a altura 1080; letterbox de Ultralytics | decode → resize `INTER_AREA` a 640×360 → letterbox 640×640 (pad 114, `INTER_LINEAR`) |
| 8 | **Versión Ultralytics** | 8.4.22 (venv de IAModel en wudebian) | **8.4.42** (`pyproject.toml` de C1) |
| 9 | **Export** | `conf kp ≤ 0.5 → NaN`, `valid_mask` con ≥ 5 KP | Igual **solo si** `CONTINUOUS_POSE_EXPORT_IAMODEL_FULL=true`; el YAML por defecto es 0.35 / 2 KP con gap-fill `interpolate` |
| 10 | **Unidad de salida** | Un `.npy` por clip/persona | Chunks de 8 s con stride 4 s por sesión (máx. 120 s de sesión) |

### 2.2 El hallazgo del canal de color (prioridad máxima)
C1 confirma que no hay conversión BGR→RGB antes del tensor de Triton:

```python
# kanvis_monitoring/pose/triton/preprocess.py (C1)
def chw_normalized(boxed: np.ndarray, *, input_fp16: bool) -> np.ndarray:
    if input_fp16:
        return _U8_TO_UNIT_FP16[np.ascontiguousarray(boxed.transpose(2, 0, 1))]
    return np.ascontiguousarray(boxed.transpose(2, 0, 1).astype(np.float32) / 255.0)
```
OpenCV entrega BGR. Ultralytics convierte a RGB en su preproceso (`im[..., ::-1]`) y el grafo ONNX/TRT exportado **no** lo hace.
Si el ONNX de C1 no inserta un flip (**[VERIFICAR]**), el modelo recibe R y B intercambiados.
Efectos esperables (hipótesis): confianzas de keypoints más bajas → más NaN con el umbral 0.5 → menos frames con ≥ 5 KP → `valid_mask` distinto del de entrenamiento.
**Consecuencia operativa:** corregirlo cambia la línea base de FP; hay que medir una nueva tras el fix y revisar umbrales.

### 2.3 Otros riesgos señalados
- ID por orden izquierda→derecha cuando falta ID del tracker: con varias personas el esqueleto de una sesión podría saltar entre personas. **[VERIFICAR con dumps]**.
- Dos versiones de Ultralytics distintas entre entrenamiento y producción.
- El tracker de producción es *stateful* por cámara y las sesiones usan reloj UTC; el offline debe inyectar tiempo sintético (`frame_idx / fps`).
- El plan TensorRT **no es portable** entre arquitecturas de GPU ni versiones de TensorRT: se construye por máquina desde el mismo ONNX.
- El `.engine` de Ultralytics **no** sirve como `model.plan` de Triton (magic 833 + cabecera propia → Triton no deserializa). Ya ocurrió en el piloto.

---

## 3. Decisiones

### 3.1 Tomadas
| ID | Decisión |
|---|---|
| D1 | **Producción es la referencia.** El extractor offline replica a C1, no al revés (cambiar producción obliga a revalidar todo el sistema). |
| D2 | El proyecto nuevo es un **repo separado** con dos entregables: imagen/modelo Triton y paquete `pose_core`. |
| D3 | El código del camino caliente de C1 (`pipeline.py`, `triton_pool.py`, batch collector, SHM, workers de preproceso) **se queda en C1**. |
| D4 | El plan TRT no se versiona en git; se construye desde el ONNX (hash registrado). |
| D5 | Un `pose_bundle_id` (+ hashes) acompaña a todo artefacto generado. |
| D6 | Ultralytics se fija a **una sola versión** en todo el sistema (propuesta: la de C1, 8.4.42). |

### 3.2 Pendientes (el agente debe preparar la medición, la decisión es humana)
| ID | Pregunta | Cómo se decide |
|---|---|---|
| P1 | ¿Se modifica C1 para convertir BGR→RGB? | Tras el A/B de §10.1 y confirmación de C1 |
| P2 | ¿Tracker "off" en el offline para el grueso de clips? | Solo si el test de equivalencia de §10.3 lo permite |
| P3 | ¿Re-extraer todo el corpus de entrenamiento con este pipeline? | Tras el test de deriva de §10.4 |
| P4 | ¿Se mantiene ByteTrack en producción o se cambia? | No cambiar sin evidencia; fuera del alcance inicial |

---

## 4. Arquitectura

```text
                    ┌─────────────────────── kanvis-pose (este repo) ───────────────────────┐
                    │  model/        export ONNX (dynamic, batch 32, end2end) + build plan    │
                    │  triton/       Dockerfile + config.pbtxt.template                       │
                    │  pose_core/    preprocess · postprocess · tracker · export · bundle     │
                    │                clip_extractor · cli                                     │
                    │  tests/golden  clip conocido + resultados esperados                     │
                    └───────────┬─────────────────────────────────────────┬──────────────────┘
        imagen Triton + pose_core│                                          │ imagen Triton + pose_core
                                 ▼                                          ▼
          C1 (producción, modo stream)                       IAModel (offline, modo clip)
          RTSP → colas → pose_core primitives                vídeo → pose_core.ClipExtractor
          → Triton (mismo plan) → MinIO                      → Triton (mismo plan) → data_result/
```

- **Triton** es una *instancia* desplegada, no el proyecto. En producción sigue corriendo el de C1. El offline se conecta al mismo (de noche) o a una **segunda instancia** con otro `config.pbtxt` (más batching) y el **mismo plan**.
- **`pose_core`** ofrece primitivas sin estado global ni reloj de pared, más un `ClipExtractor` para offline.

### 4.1 Estructura del repo (propuesta)
```text
kanvis-pose/
├── pyproject.toml                 # paquete pose_core (Python >=3.11); extra [yolo] para ultralytics fijado
├── README.md
├── model/
│   ├── export_onnx.py             # basado en scripts/export_triton_pose_onnx.py de C1
│   ├── build_plan.sh              # basado en scripts/triton_pose_entrypoint.sh de C1
│   └── config.pbtxt.template      # basado en scripts/triton_pose_config.py de C1
├── triton/
│   └── Dockerfile                 # FROM nvcr.io/nvidia/tritonserver:25.02-py3 (+ trtexec si hace falta)
├── pose_core/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── postprocess.py
│   ├── triton_client.py
│   ├── tracker.py
│   ├── export.py
│   ├── bundle.py
│   ├── clip_extractor.py
│   └── cli.py
├── tests/
│   ├── golden/                    # clips + esperados (ver §10.2)
│   ├── test_preprocess.py
│   ├── test_postprocess.py
│   ├── test_export_contract.py
│   └── test_clip_extractor.py
├── scripts/
│   ├── ab_rgb_bgr_onnx.py         # experimento A.0 (§10.1)
│   ├── compare_with_iamodel.py    # deriva vs pose_extractor_clean (§10.4)
│   └── bench_batch_determinism.py # determinismo del batch (§10.5)
├── reference/                     # copias de solo lectura de los otros repos (ver Anexo D)
└── docs/
```

---

## 5. Contratos

### 5.1 Contrato de keypoints
- Se usan **8 keypoints COCO**: `KEEP_KPS = [5, 6, 7, 8, 9, 10, 11, 12]` (hombros, codos, muñecas, caderas; sin cabeza ni piernas).
- Coordenadas **normalizadas** `xyn` en `[0, 1]` respecto al **frame de inferencia** (x/ancho, y/alto). `y = 0` arriba. Sin invertir.
- Confianza de keypoint `≤ MIN_KP_CONF = 0.5` → coordenada `NaN`.
- **Frame con cuerpo visible**: al menos `BODY_VISIBLE_MIN_KPS = 5` keypoints sin NaN.
- `valid_mask[t] = (visibles[t] >= 5)`.
- Un track ausente en un frame **no omite la fila**: se escribe `NaN` en `poses_full` y fila vacía en `keypoint_mask` (1:1 entre frame y pose).

### 5.2 Artefactos por persona (layout compatible con el entrenamiento)
```text
data_result/{categoria}/{clip_name}/
  meta.json
  clip.mp4                       (opcional; copia del clip procesado)
  user_{track_id}/
    poses_full.npy   [T, 8, 2] float32, NaN donde conf <= 0.5 o track ausente
    poses.npy        [T', 8, 2] float32, solo frames con valid_mask True
    keypoint_mask.npy [T, 8]   bool, True por keypoint visible
    valid_mask.npy   [T]       bool, True si >= 5 keypoints visibles
```
Añadir en el modo del proyecto nuevo, **sin romper** a los consumidores actuales:
- `pose_bundle.json` por clip, y los campos `pose_bundle_id` y `yolo_backend` en `meta.json`.
- Opcional: `pose_frame_timestamps.npy` `[T]` float64 en segundos (`frame_idx / fps` en offline).

`meta.json` de clip (campos que ya existen hoy; ver Anexo A.4):
`clip_name, fps, frame_count, video_frame_count, min_valid_frames, cat, clip_duration, yolo_model, yolo_backend, users[]`.
Cada usuario: `track_id, valid_pct, rel, valid_frames, total_frames, poses_full_count, poses_filtered_count, passes_filters, keypoint_confidence_avg, occlusion_ratio, bbox_aspect_ratio, subject_velocity, clothes{top,bottom}, user_cat, is_primary_robber (solo cat 6)`.
`yolo_backend` pasa de `"engine"|"pt"` a `"triton_tensorrt_plan"`.

### 5.3 Qué hace el entrenamiento con esos ficheros (para no romperlo)
`PoseDataset.__getitem__` (Anexo A.5): carga `poses_full.npy`; si existe `valid_mask.npy`, **elimina** los frames con `False`; `NaN → 0`;
luego `normalize_sequence` (centro por frame = media de joints; **escala global** = media de distancias joint-centro sobre todos los frames),
`add_velocity` (diff con `prepend`), `temporal_resize` a `seq_len` (submuestreo `linspace` o padding repitiendo el último frame).
**Consecuencia:** el consumidor necesita `poses_full.npy` y `valid_mask.npy` con exactamente la semántica de §5.1.

### 5.4 Contrato de producción (C1) a replicar en modo stream
Chunks de `window_seconds = 8` con `stride_seconds = 4`, por `session_id`. Por chunk: `poses_full.npy`, `valid_mask.npy`,
`keypoint_mask.npy`, `pose_frame_timestamps.npy` (epoch UTC), `meta.json`, `diag.json`, `pose_bundle.json`.
Perfil IAModel en MinIO: conf KP 0.5, ≥ 5 KP, gap-fill `none`, frames `is_synthetic` marcados `valid_mask = False`.

---

## 6. Especificación de `pose_core`

### 6.1 Principios
1. **Sin reloj de pared.** Todo timestamp entra como parámetro (`ts`), nunca `time.time()` ni `datetime.now()`.
2. **Sin estado global.** El estado del tracker vive en una instancia por stream/clip.
3. **Sin dependencias de MinIO, colas, RTSP ni del contrato `InteractionChunk` de C1.**
4. **Cero copias innecesarias** en preproceso y decode (C1 los usa en un camino caliente; medir antes de integrar).
5. **Un solo lugar** para el orden de canales: parámetro explícito `input_color="bgr"` y conversión a RGB dentro de `to_tensor` (configurable para el A/B).

### 6.2 API propuesta (firmas orientativas)
```python
# preprocess.py
@dataclass(frozen=True)
class PreprocessConfig:
    infer_max_width: int = 640          # C1: pose.infer_frame_max_width
    infer_max_height: int = 360         # C1: pose.infer_frame_max_height
    downscale_interpolation: int = cv2.INTER_AREA
    imgsz: int = 640                    # C1: pose.tracking_imgsz
    pad_value: int = 114
    letterbox_interpolation: int = cv2.INTER_LINEAR
    input_fp16: bool = True
    input_color: Literal["bgr", "rgb"] = "bgr"   # color del frame de entrada
    model_color: Literal["rgb", "bgr"] = "rgb"   # color que espera el modelo (Ultralytics = rgb)

def prepare_infer_image(frame_bgr: np.ndarray, cfg: PreprocessConfig) -> np.ndarray: ...   # downscale si excede max
def letterbox(img: np.ndarray, cfg: PreprocessConfig) -> tuple[np.ndarray, LetterboxMeta]: ...
def to_tensor(boxed: np.ndarray, cfg: PreprocessConfig) -> np.ndarray: ...                 # CHW /255 (+ BGR->RGB si procede)

# postprocess.py
@dataclass
class Detection:
    bbox_xyxy: np.ndarray   # float32 [4], en coords del frame de inferencia
    conf: float
    kpts: np.ndarray        # float32 [17, 3] (x, y, conf), en coords del frame de inferencia
def decode_end2end(raw: np.ndarray, meta: LetterboxMeta, conf_threshold: float) -> list[Detection]: ...
# raw: (300, 57) por frame = [x1,y1,x2,y2,conf,cls] + 17*(x,y,conf), en espacio letterbox.

# triton_client.py
class TritonPoseClient:
    def __init__(self, url: str, model_name: str = "yolo26s-pose", input_fp16: bool = True,
                 use_system_shm: bool = False): ...
    def infer(self, batch: np.ndarray) -> np.ndarray: ...   # (B,3,640,640) -> (B,300,57)

# tracker.py
class TrackerAdapter:
    def __init__(self, kind: Literal["bytetrack", "botsort", "none"], config_path: str | None,
                 frame_rate: float): ...
    def update(self, detections: list[Detection], image: np.ndarray) -> list[TrackedDetection]: ...
    def reset(self) -> None: ...
# Para bytetrack/botsort usar las clases de Ultralytics (ultralytics.trackers) con la versión fijada.
# Para "none": devolver las detecciones sin ID, y dejar la selección de persona a clip_extractor.

# export.py
KEEP_KPS = [5, 6, 7, 8, 9, 10, 11, 12]
def build_masked_pose(kpts17: np.ndarray, min_kp_conf: float = 0.5) -> np.ndarray: ...     # [8,2], NaN si conf<=min
def keypoint_mask(pose8: np.ndarray) -> np.ndarray: ...                                    # [8] bool
def frame_valid(pose8: np.ndarray, min_visible: int = 5) -> bool: ...
def write_person_artifacts(user_dir: Path, rows: PersonRows) -> None: ...                  # §5.2

# bundle.py
def build_pose_bundle(onnx_path, plan_path, pt_path, triton_image_tag, preprocess_cfg,
                      tracker_cfg, export_cfg) -> dict: ...                                 # incluye SHA-256

# clip_extractor.py
class ClipExtractor:
    def __init__(self, client: TritonPoseClient, preprocess: PreprocessConfig, tracker: TrackerAdapter,
                 bundle: dict, decode_conf: float = ...): ...
    def extract(self, video_path: str, fps_override: float | None = None) -> ClipPoseResult: ...
```

### 6.3 Reglas del `ClipExtractor` (modo offline)
- Lee **todos los frames** del vídeo (sin submuestreo); `ts = frame_idx / fps`; sin hold-last; sin sesiones ni chunks.
- Procesa cada clip como un **stream independiente** con su propio tracker. Para rendimiento, permite **varios clips en paralelo** (cada uno con su estado) y agrupa frames en lotes hacia Triton.
- **Modo `tracker=none`**: elige la persona por regla explícita y versionada (propuesta: la detección de mayor presencia, con continuidad por IoU frame a frame). Solo se activa si pasa el test de §10.3.
- **Modo `tracker=bytetrack`** (por defecto): misma configuración que producción (`bytetrack_long.yaml`, Anexo B).
- Salida compatible con §5.2. Por defecto escribe todos los tracks que cumplan los filtros actuales (`BODY_VISIBLE_MIN_FRAMES = 5`, `BODY_VISIBLE_MIN_RATIO = 0.2`), igual que el extractor actual.
- Opción `--single-user-only` equivalente a la del extractor actual (descarta el clip si hay más de un track).

### 6.4 Umbral de detección en offline
Hay que **fijar y registrar** el umbral efectivo que llega al tracker. En producción, sesión activa: decode `max(0.05, 0.18×0.5) = 0.09`, y luego filtro `0.18`.
Propuesta: el modo clip replica estos valores por defecto (parámetro explícito `decode_conf`, `parse_conf`) y los guarda en `pose_bundle.json`.

---

## 7. Modelo y Triton

### 7.1 Export ONNX (replicar el de C1)
```python
# scripts/export_triton_pose_onnx.py de C1 (parámetros)
model.export(format="onnx", imgsz=640, dynamic=True, simplify=True, opset=17,
             batch=32, end2end=True)     # ultralytics==8.4.42
```
- Salida end2end: `(batch, 300, 57)` (top-300, sin NMS externo). Entrada `images` `[3,640,640]`, `FORMAT_NCHW`.
- Fuente de pesos: `yolo26s-pose.pt` (registrar SHA-256).

### 7.2 Plan TensorRT (replicar `triton_pose_entrypoint.sh` de C1)
```text
trtexec --onnx=model.onnx --saveEngine=model.plan \
        --minShapes=images:1x3x640x640 --optShapes=images:16x3x640x640 --maxShapes=images:32x3x640x640 \
        --fp16 --inputIOFormats=fp16:chw --timingCacheFile=trt_timing.cache
```
- Imagen: `nvcr.io/nvidia/tritonserver:25.02-py3` (el binario `trtexec` debe estar disponible: C1 lo usa dentro de su sidecar; comprobar. La imagen `nvcr.io/nvidia/tensorrt:25.02-py3` también lo trae).
- GPU de producción piloto: RTX 5090. **El plan no es portable**: reconstruir en cada máquina.
- **No** usar el `.engine` de Ultralytics como `model.plan`.

### 7.3 `config.pbtxt` (valores piloto de C1)
| Campo | Valor |
|---|---|
| `max_batch_size` | 32 |
| `dynamic_batching.max_queue_delay_microseconds` | 3000 |
| `preferred_batch_size` | 4, 8, 16, 32 |
| `preserve_ordering` | true |
| `instance_group` | 2 instancias en GPU |
| Entrada `data_type` | `TYPE_FP16` (si `input_fp16`) |
| Salida | `TYPE_FP32` |
| Backend | `tensorrt_plan`; modelo `yolo26s-pose`; gRPC en puerto 8011 |

El modo offline puede usar una **segunda instancia** con otro `config.pbtxt` (batches mayores, `max_queue_delay` más alto), **mismo plan**.

### 7.4 Imagen Triton versionada
`kanvis-triton-pose:<pose_bundle_id>`: Triton + model repository con el plan construido desde el ONNX del repo, o bien una imagen que construye el plan al arrancar si cambia el ONNX (como hace hoy C1). Documentar cuál se elige; en cualquier caso registrar `plan_sha256` en el bundle.

---

## 8. Dos modos de uso

| | Modo stream (C1) | Modo clip (offline) |
|---|---|---|
| Entrada | RTSP, colas, backlog | fichero de vídeo |
| Tiempo | reloj UTC + timestamps de frame | `frame_idx / fps` |
| Frames | ~15 fps en sesión; hold-last posible | todos |
| Tracker | ByteTrack por cámara | ByteTrack por clip (o `none` si se valida) |
| Salida | chunks 8 s / stride 4 s en MinIO | `data_result/` por clip |
| Dónde vive el código | C1 (usa primitivas de `pose_core`) | `pose_core.ClipExtractor` |

---

## 9. Fases de desarrollo

Cada fase tiene criterios de aceptación verificables. No pasar a la siguiente sin cumplirlos.

### Fase 0 — Arranque del repo
- Crear el repo, `pyproject.toml`, CI mínima (lint + tests), pin de versiones (Python ≥ 3.11, `ultralytics==8.4.42`, `tritonclient[grpc]`, `numpy`, `opencv-python-headless`).
- Traer los ficheros de §Anexo D a `reference/`.
- **Aceptación:** `pip install -e .` funciona; CI verde; `reference/` documentado.

### Fase A.0 — Experimento RGB vs BGR (independiente de C1)
Ver §10.1. Puede hacerse **desde el primer día**, con solo el ONNX y vídeos.
- **Aceptación:** informe con distribución de confianza de keypoints, % de keypoints > 0.5, % de frames con ≥ 5 KP y diferencia de `valid_mask` entre RGB y BGR sobre ≥ 100 clips. Conclusión escrita sobre si hay efecto apreciable.

### Fase 1 — Modelo e imagen Triton reproducibles
- `model/export_onnx.py`, `model/build_plan.sh`, `model/config.pbtxt.template`, `triton/Dockerfile`, `pose_core/bundle.py`.
- Smoke contra Triton (equivalente a `check_triton_pose_batch.py` de C1): batch 1/4/16/32 con tensor 1×3×640×640.
- **Aceptación:** `docker build` reproduce la imagen; Triton `READY`; smoke OK; `pose_bundle.json` generado con SHA-256 de ONNX, plan y `.pt`; el plan es **TRT crudo** (primer u32 ≠ 833).

### Fase 2 — Primitivas de `pose_core` con tests de paridad
- Portar `preprocess.py`, `postprocess.py`, `triton_client.py` y `export.py` desde C1, con el parámetro de color explícito (§6.2).
- Tests unitarios: letterbox (pad 114, `INTER_LINEAR`), `prepare_infer_image` (`INTER_AREA`), decode end2end, reglas de §5.1.
- **Paridad con C1:** dado un conjunto de frames, las salidas de `pose_core` (con `input_color="bgr"`, `model_color="bgr"` para reproducir C1 tal cual) deben coincidir con las de C1 dentro de tolerancia (*propuesta inicial*: error absoluto máximo de keypoints normalizados ≤ 0.005 y de confianza ≤ 0.02; calibrar con la medición de §10.5).
- **Aceptación:** tests verdes; informe de paridad con C1; benchmark de preproceso+decode sin regresión respecto al código actual de C1 (*propuesta*: ≤ 3 % más lento).

### Fase 3 — Tracker y `ClipExtractor` + CLI
- `tracker.py` con ByteTrack (config de producción) y BoT-SORT (config de entrenamiento) para poder comparar; modo `none`.
- `clip_extractor.py` y `cli.py`: `kanvis-pose extract-clips --video-dir ... --out ... --triton-url ... --tracker bytetrack`.
- Salida compatible con §5.2; `pose_bundle.json` por clip.
- **Aceptación:** sobre los clips golden (§10.2) se generan los artefactos con el layout correcto; los consumidores actuales de IAModel (`build_clip_index.py`, carga de `PoseDataset`) leen la salida **sin modificaciones**.

### Fase 4 — Paridad y deriva
Ver §10.3 y §10.4.
- **Aceptación:** informe comparando (a) `pose_core` ↔ `pose_extractor_clean.py` (Ultralytics), (b) con/sin tracker, (c) RGB/BGR, sobre 100–200 clips de validación, con métricas de §10.4. Recomendación escrita sobre P2 y P3.

### Fase 5 — Integración en C1 (la hace C1 con apoyo de este repo)
Ver §11. **Aceptación:** C1 usa `pose_core` en las primitivas, sin regresión de throughput (medido con N cámaras sostenidas), y el chunk de prueba con `pose_bundle_id` cruza con el extractor.

### Fase 6 — Integración en IAModel
Ver §12. **Aceptación:** `pose_extractor_clean.py` reemplazado por el CLI en el flujo de extracción; dataset v2 generado con `pose_bundle_id` en `meta.json`; plan de reentrenamiento decidido (P3).

### Fase 7 — Operación del Triton compartido
- Runbook: levantar una segunda instancia para offline; ejecutar de noche; límites de GPU para no pisar producción.
- Requisito de SHM: el cliente offline debe estar en el mismo host (o `ipc: host`), o usar gRPC con `use_system_shm=False`.
- **Aceptación:** extracción nocturna de N clips sin afectar a las métricas de latencia de producción.

---

## 10. Tests y experimentos

### 10.1 A.0 — A/B de canales RGB vs BGR (`scripts/ab_rgb_bgr_onnx.py`)
1. Entrada: ONNX (`dynamic`, end2end), carpeta de vídeos, N frames por clip (p. ej. 20 uniformes).
2. Para cada frame: preprocesar con letterbox 640 (pad 114) y ejecutar con **onnxruntime** dos veces: tensor RGB y tensor BGR.
3. Métricas por variante: conf de detección de persona, conf de los 8 keypoints, % de keypoints con conf > 0.5, nº de keypoints visibles por frame, % de frames con ≥ 5 visibles, desplazamiento de coordenadas entre variantes.
4. Salida: CSV + resumen (media, mediana, p5/p95) y gráfica de distribución.
Se puede ejecutar en CPU. **No requiere** Triton ni C1.

### 10.2 Test golden
**Principio: el golden es el VÍDEO, no los `.npy`.** Un `.npy` solo es comparable con otro si salió del **mismo vídeo**.
Los clips de entrenamiento (`data_result`) y las sesiones de producción (MinIO) son de personas y momentos distintos, y no se pueden cruzar entre sí.
Cada pipeline genera su salida sobre el mismo vídeo y las referencias se guardan por etiqueta de pipeline:

| Etiqueta | Pipeline | Cómo se obtiene sobre el golden |
|---|---|---|
| **E1** | `pose_extractor_clean.py` (IAModel, Ultralytics) | Ya existe si el vídeo sale de `data_result`; si no, ejecutar el extractor sobre el vídeo |
| **E2** | `pose_core` (este repo) | Se genera con `ClipExtractor` |
| **E3** | C1 (Triton + tracker de producción) | Hay que **pasar el vídeo a C1** (artefacto de prueba: `simulate_continuous_session.py` o dumps de depuración de §11.7) |

**Layout:**
```text
tests/golden/{nombre_del_clip}/
  clip.mp4                   # el vídeo (Git LFS o ruta externa en KANVIS_GOLDEN_DIR)
  e1_iamodel/                # meta.json + user_*/poses_full.npy, valid_mask.npy, poses.npy, keypoint_mask.npy
  e2_pose_core/              # salida aceptada de pose_core (se congela tras validarla; es la de regresión)
  e3_c1/                     # chunks + debug_detections.jsonl + debug_tracks.jsonl (cuando existan)
```

**Dos orígenes posibles del vídeo:**
- **Opción A (recomendada): clip curado de entrenamiento** (`data_result/.../clip.mp4`). E1 ya existe; E3 requiere que C1 procese el vídeo (su pipeline está pensado para RTSP en vivo: hold-last, fps de inferencia ~15; esa diferencia es parte de lo que se mide).
- **Opción B: sesión de producción con vídeo grabado** (`session_video_from_rolling_buffer`, `capture_record_full_stream`). E3 ya existe (chunks de 8 s, stride 4 s, en MinIO); hay que ejecutar E1 y E2 sobre el vídeo grabado y alinear chunks con frames mediante timestamps. Útil para validar con varias personas e ID switches; el vídeo grabado puede estar recodificado.
Usar la Opción A para el golden y los tests de regresión; reservar la B para una validación posterior sobre 10–20 sesiones reales.

**Qué comprueban los tests:**
- Regresión (CI): `ClipExtractor` sobre `clip.mp4` debe reproducir `e2_pose_core/` (forma, dtype, semántica de NaN, `valid_mask`) dentro de tolerancia.
- Deriva (Fase 4, §10.4): comparar E1 ↔ E2 ↔ E3 sobre el mismo vídeo.
- Se incluye también el chunk de producción de muestra `SAMPLE_production_chunk_mergedfps.npy` (`[60, 8, 2]`, 8,1 % de NaN, frames 55–59 idénticos) como caso patológico para el export.

### 10.3 Equivalencia con/sin tracker (decide P2)
Sobre 100–200 clips curados, comparar `tracker=bytetrack` y `tracker=none`: frames totales, `valid_mask.sum()`, frames perdidos al inicio, nº de tracks y `p_robbery` del ensemble 41/48 (ver §12).
Criterio *propuesta*: si la mediana de `|Δp_robbery|` es < 0.02 y ≥ 99 % de los clips mantienen el mismo `valid_mask` útil, "none" es aceptable para el grueso.

### 10.4 Test de deriva (decide P3)
Sobre los mismos clips, comparar tres extracciones: **(E1)** `pose_extractor_clean.py` actual, **(E2)** `pose_core` con perfil de producción, **(E3)** C1 real (cuando existan los dumps).
Métricas: frames válidos, `valid_mask.sum()`, nº de tracks, saltos de ID, y `p_robbery` del ensemble. Informar los clips con mayor diferencia.
Si el delta de scores es pequeño → basta con registrar el bundle y no re-extraer. Si es grande → re-extraer el corpus con `pose_core` y reentrenar.

### 10.5 Determinismo del batch en Triton
Mismo frame inferido solo y dentro de batches 4/16/32 (`preserve_ordering: true`). Registrar el error absoluto máximo de keypoints y confianza. Sirve para calibrar las tolerancias de §9 Fase 2.

### 10.6 Test de causalidad (modo stream)
Añadir frames posteriores a un stream no debe cambiar salidas ya emitidas. No hay normalización global con frames futuros en el export.

---

## 11. Integración con C1 (`kanvis-monitoring`)

**Quién:** el equipo C1, con apoyo del agente de este repo. **Qué NO se cambia:** `pipeline.py`, `triton_pool.py`, batch collector F7, SHM, workers de preproceso.

Pasos:
1. **Versionado:** añadir `pose_core` como dependencia fijada por tag (p. ej. `kanvis-pose @ git+...@vX.Y.Z`). Actualizar de forma deliberada, no automática.
2. **Modelo y Triton:** sustituir su export ONNX y su construcción de plan por los de este repo, o garantizar que producen exactamente el mismo ONNX (comparar SHA-256) y los mismos flags de `trtexec`. Mantener `triton_max_batch_size 32`, `instance_group 2`, FP16.
3. **Preproceso:** importar `prepare_infer_image`, `letterbox`, `to_tensor` de `pose_core`. Validar paridad (§9 Fase 2) **antes** de cambiar la conversión de color.
4. **Corrección BGR→RGB (decisión P1):** activar `model_color="rgb"` en un entorno de prueba; comparar con el estado actual; si procede, desplegar y **medir una nueva línea base** de FP/sesión durante 3–7 días antes de ajustar umbrales o reentrenar.
5. **Postproceso, tracker y export:** importar `decode_end2end`, `TrackerAdapter` y `export` de `pose_core`. El tracker sigue siendo por cámara; C1 conserva la gestión de sesiones (`track_sessions.py`).
6. **Perfil de export:** `CONTINUOUS_POSE_EXPORT_IAMODEL_FULL=true`, `CONTINUOUS_POSE_EXPORT_MIN_KP_CONF=0.5`, `CONTINUOUS_POSE_EXPORT_MIN_VISIBLE_KPS=5`, gap-fill `none`, `KANVIS_POSE_BUNDLE_ID=<bundle>`.
7. **Dumps de depuración** (para la Fase 4): implementar `debug_detections.jsonl` y `debug_tracks.jsonl` con el esquema siguiente (una línea por frame y cámara).
   ```json
   {"frame_idx": 0, "ts": 1760000000.0, "cam": "cam1", "is_synthetic": false,
    "infer_size": [640, 360], "letterbox": {"scale": 1.0, "pad": [0, 140]},
    "dets": [{"bbox_xyxy": [0,0,0,0], "conf": 0.0, "kpts": [[0.0, 0.0, 0.0]]}]}
   ```
   ```json
   {"frame_idx": 0, "cam": "cam1",
    "tracks": [{"track_id": 3, "bbox_xyxy": [0,0,0,0], "score": 0.0, "state": "tracked"}],
    "exported_row": {"track_id": 3, "pose_8x2": [[0.0, 0.0]], "kp_conf_8": [0.0], "valid": true}}
   ```
8. **Gate de rendimiento:** medir throughput (cámaras sostenidas) y latencia antes y después; si hay regresión, revertir al código propio de C1 para esa pieza.

Datos que C1 aún debe aportar: ver §14.

---

## 12. Integración con IAModel (entrenamiento)

1. Instalar `pose_core` en el entorno de extracción (mismo `ultralytics` fijado).
2. Sustituir `pose_extractor_clean.py` por `kanvis-pose extract-clips` en el flujo de extracción. El script actual se conserva como referencia/herramienta de exploración.
3. **Compatibilidad:** la salida debe leerse sin cambios con `build_clip_index.py`, `PoseDataset` y las rutinas de evaluación (`poses_full.npy` + `valid_mask.npy` por usuario, `meta.json` por clip). `config.py` (rutas `PATH_ROOTS`, `OUTPUT_BASE`, `CLIP_SCALE_HEIGHT`) sigue gobernando entrada y salida.
4. **Filtro por bundle:** los datos de entrenamiento futuros se seleccionan por `pose_bundle_id`. Los `.npy` antiguos se consideran "v1 / bundle desconocido" y no se mezclan sin saberlo.
5. **Uso nocturno:** extraer de noche contra el Triton compartido o una segunda instancia; entrenar después (el entrenamiento del modelo de movimiento no necesita Triton, solo GPU).
6. **Ensemble actual para las métricas de §10:** modelos #41 (`seq_len` 64) y #48 (`seq_len` 96) de la campaña de octubre, AND con umbral 0.80, sobre `poses_full.npy` (con `valid_mask`). Las rutas de checkpoints las aporta IAModel.
7. Reentrenar solo tras decidir P3.

---

## 13. Riesgos

| Riesgo | Mitigación |
|---|---|
| Regresión de rendimiento en C1 al usar `pose_core` | Gate de §11.8; primitivas sin copias; benchmark previo |
| El fix BGR→RGB cambia el comportamiento de producción | Probar en entorno de prueba; nueva línea base de FP antes de tocar umbrales |
| Plan TRT incompatible entre GPUs | Construir por máquina desde el ONNX; no versionar el plan |
| Diferencias de versión de Ultralytics | Fijar una versión en `pyproject.toml` |
| Tracker offline distinto del de producción | Configuración idéntica; test de §10.3 |
| Cambios del repo compartido rompen producción | Versionado por tag; C1 actualiza de forma deliberada; tests golden en CI |
| Ambigüedad sobre quién es dueño del repo | Definir responsable de aprobar cambios que afecten a producción y a entrenamiento |

---

## 14. Datos pendientes (C1 aún no los ha entregado)

| Dato | Estado |
|---|---|
| Estadísticas reales de `is_synthetic`, fps efectivo y huecos de timestamps en chunks con `IAMODEL_FULL=true` | Sin medir; C1 se comprometió a un script de auditoría |
| Determinismo del batch en Triton (§10.5) | Sin medir |
| Dumps `debug_detections` / `debug_tracks` | No implementados (estimación C1: +3–5 días laborables) |
| SHA-256 del ONNX y del plan en producción | Se generarán en `pose_bundle.json` tras el despliegue |
| Confirmación de si el ONNX de C1 inserta flip de canales | **[VERIFICAR]** |
| Hardware y memoria libre de GPU en ventana nocturna | Sin medir |
| Resolución nativa por cámara y si todas son 16:9 | La define el catálogo/NVR; no fijada en C1 |

---

## Anexo A — Código de referencia (IAModel)

### A.1 Constantes y reglas de pose (`pose_extractor_clean.py`)
```python
KEEP_KPS = [5, 6, 7, 8, 9, 10, 11, 12]       # hombros, codos, muñecas, caderas
MIN_KP_CONF = 0.5                              # conf <= 0.5 -> NaN
BODY_VISIBLE_MIN_KPS = 5                       # mínimo de keypoints visibles por frame
BODY_VISIBLE_MIN_FRAMES = 5
BODY_VISIBLE_MIN_RATIO = 0.2
RELIABILITY_THR = 0.9
MIN_COVERAGE_RATIO = 0.8
DEFAULT_FPS = 12                               # solo si no se puede leer el fps del vídeo
OCCLUSION_CONF_THR = 0.3
CRITICAL_KPS = [7, 8, 9, 10]
NAN_POSE = np.full((len(KEEP_KPS), 2), np.nan, dtype=np.float32)
EMPTY_KP_MASK = np.zeros(len(KEEP_KPS), dtype=bool)

def _build_masked_pose(kpts_row, confs_row):
    out = kpts_row[np.array(KEEP_KPS, dtype=int)].astype(np.float32, copy=True)
    for j, coco_idx in enumerate(KEEP_KPS):
        if float(confs_row[coco_idx]) <= MIN_KP_CONF:
            out[j] = np.nan
    return out

def _keypoint_mask_from_pose(kpt_pose):          # [8] bool
    return ~np.isnan(kpt_pose).any(axis=1)

def _visible_kp_count(kpt_pose):
    return int(_keypoint_mask_from_pose(kpt_pose).sum())

def _frame_body_visible(kpt_pose):
    return _visible_kp_count(kpt_pose) >= BODY_VISIBLE_MIN_KPS

def _frame_usable(kpt_pose):
    if np.isnan(kpt_pose).all():
        return False
    return _frame_body_visible(kpt_pose)

def _save_user_pose_files(user_dir, info):
    poses_full_arr = np.array(info["poses_full"], dtype=np.float32)
    np.save(user_dir / "poses_full.npy", poses_full_arr)
    np.save(user_dir / "poses.npy", np.array(info["poses"], dtype=np.float32))
    keypoint_mask_arr = np.array(info["keypoint_masks"], dtype=bool)
    np.save(user_dir / "keypoint_mask.npy", keypoint_mask_arr)
    visible_per_frame = np.sum(keypoint_mask_arr, axis=1)
    valid_mask = visible_per_frame >= BODY_VISIBLE_MIN_KPS
    np.save(user_dir / "valid_mask.npy", valid_mask)
```

### A.2 Bucle de seguimiento sobre un clip (comportamiento a reproducir)
```python
results = track_model.track(source=clip_path, tracker=custom_tracker_yaml, persist=True,
                            verbose=False, stream=True, device=DEVICE, half=True)   # conf=0.1 por defecto en track()
for r in results:
    if frame is None or r.keypoints is None or r.boxes.id is None:
        # frame sin detecciones/ID: para cada track conocido añadir NAN_POSE y EMPTY_KP_MASK, total += 1
        continue
    ids = r.boxes.id.int().tolist(); kpts = r.keypoints.xyn; confs = r.keypoints.conf
    for tid in tracks_conocidos:
        if tid not in ids: añadir NAN_POSE / EMPTY_KP_MASK; total += 1
    for i, tid in enumerate(ids):
        kpt_pose = _build_masked_pose(kpts[i], confs[i])
        poses_full.append(kpt_pose); keypoint_masks.append(_keypoint_mask_from_pose(kpt_pose))
        if _frame_usable(kpt_pose): poses.append(kpt_pose)
        total += 1
```
Nota: este bucle **descarta** frames sin `boxes.id`. En producción no se descartan. `pose_core` debe permitir ambos comportamientos como parámetro y registrar cuál se usó.

### A.3 Lectura de tiempo del clip
`fps` y duración se leen con `cv2.VideoCapture` (`CAP_PROP_FPS`, `CAP_PROP_FRAME_COUNT`); si el fps no es válido se usa `DEFAULT_FPS = 12`.
Los clips de entrenamiento son de 15 fps (confirmado por el responsable de datos).
Escalado de vídeo: `ffmpeg ... -vf scale=-2:{CLIP_SCALE_HEIGHT}` con `CLIP_SCALE_HEIGHT = 1080`.

### A.4 Estructura de `meta.json` por clip
Ver §5.2 (lista de campos). Los clips de categoría 6 reciben `is_primary_robber = (índice de usuario == 0)` si no viene previo.

### A.5 Preproceso de entrenamiento (`training/train_model_operations.py`)
```python
def normalize_sequence(poses):             # poses: [T, J, 2], coords normalizadas
    poses = poses.astype(np.float32)
    center = poses.mean(axis=1, keepdims=True)
    poses = poses - center
    scale = np.linalg.norm(poses, axis=-1).mean()   # escala GLOBAL (todos los frames y joints)
    if scale > 0:
        poses = poses / scale
    return poses

def add_velocity(poses):                    # [T, J, 2] -> [T, J, 4] (x, y, dx, dy)
    vel = np.diff(poses, axis=0, prepend=poses[0:1])
    return np.concatenate([poses, vel], axis=-1)

def temporal_resize(seq, target_len):       # submuestreo uniforme o padding repitiendo el último frame
    t = seq.shape[0]
    if t == target_len: return seq
    if t > target_len:
        idx = np.linspace(0, t - 1, target_len).round().astype(int); return seq[idx]
    pad = np.repeat(seq[-1:], target_len - t, axis=0)
    return np.concatenate([seq, pad], axis=0)

# PoseDataset.__getitem__:
poses = np.load(ex.pose_path)                       # poses_full.npy  [T, J, 2]
if ex.valid_mask_path is not None and ex.valid_mask_path.exists():
    poses = poses[np.load(ex.valid_mask_path)].copy()   # elimina frames inválidos
poses = np.nan_to_num(poses, nan=0.0, posinf=0.0, neginf=0.0)   # NaN -> 0
# ... augment opcional ...
poses = normalize_sequence(poses); poses = add_velocity(poses); poses = temporal_resize(poses, seq_len)
```
**Atención:** `NaN → 0` ocurre antes de centrar; un joint ausente se convierte en (0,0) de la imagen. Un chunk de producción que conserve frames inválidos distorsiona centro y escala (medido en la muestra: escala +31 %). Por eso es crítico que `valid_mask` se aplique igual en entrenamiento y en inferencia.

---

## Anexo B — Configuración de trackers

### B.1 Entrenamiento (`custom_tracker.yaml`, IAModel)
```yaml
tracker_type: botsort
track_high_thresh: 0.3
track_low_thresh: 0.1
new_track_thresh: 0.7
track_buffer: 300
match_thresh: 0.5
fuse_score: True
gmc_method: sparseOptFlow
with_reid: True
model: auto
proximity_thresh: 0.5
appearance_thresh: 0.25
```

### B.2 Producción (`pose/bytetrack_long.yaml`, C1)
```yaml
tracker_type: bytetrack
track_high_thresh: 0.3
track_low_thresh: 0.05
new_track_thresh: 0.25
track_buffer: 9000
match_thresh: 0.92
fuse_score: true
```

### B.3 C1 alternativo (no activo): `pose/guadia_custom_tracker.yaml`
Igual que B.1 con `model: models/yolo26n-cls.pt`. Está desactivado (`tracking.tracker_config_file: null`).

---

## Anexo C — Parámetros de C1 (de `config/monitoring.yaml` piloto)

```yaml
pose:
  inference_backend: triton
  triton_url: triton-pose:8011
  triton_model_name: yolo26s-pose
  triton_platform: tensorrt_plan
  triton_onnx_end2end: true
  triton_max_batch_size: 32
  triton_max_queue_delay_microseconds: 3000
  triton_input_fp16: true
  triton_system_shm_enabled: true
  triton_preprocess_workers: 8
  triton_gpu_instance_count: 2
  triton_server_letterbox_enabled: false
  triton_gpu_preprocess_enabled: false
  infer_frame_max_width: 640
  infer_frame_max_height: 360
  tracking_imgsz: 640
  confidence_threshold: 0.30
  iou_threshold: 0.45
  keypoint_indices_enabled: [5, 6, 7, 8, 9, 10, 11, 12]
tracking:
  algorithm: bytetrack
  tracker_config_file: null
  use_tracker_in_continuous: true
  confidence_threshold: 0.22
continuous:
  window_seconds: 8.0
  stride_seconds: 4.0
  pose_fps: 15.0
  pose_confidence_threshold_idle: 0.22
  pose_confidence_threshold_active: 0.18
  pose_gap_policy: nan
  pose_chunk_gap_fill: interpolate        # legacy; 'none' con IAMODEL_FULL
  pose_export_min_kp_conf: 0.35           # legacy; 0.5 con IAMODEL_FULL
  pose_export_min_visible_kps: 2          # legacy; 5 con IAMODEL_FULL
  session_open_max_persons_in_frame: 0    # 0 = sin límite
  session_subject_absent_seconds: 3.0
  session_exportable_gap_close_seconds: 15.0
  session_max_wall_seconds: 120.0
```
Detalles del pipeline de C1: downscale con `cv2.INTER_AREA` si el frame supera 640×360 (sin letterbox en ese paso); letterbox a 640×640 con pad 114 e interpolación lineal;
entrada FP16 NCHW `/255`; el tracker recibe un `MockYoloResult` (cajas + conf + 17 keypoints) vía `tracker.update(result, img=prepared_image)`;
`ultralytics==8.4.42`; decode end2end con umbral `max(0.05, infer_conf × 0.5)` y luego filtro a `infer_conf`.

---

## Anexo D — Ficheros a traer a `reference/` (solo lectura)

### D.1 Desde IAModel (este repo, `GuadIA-IAModel/IAModel/`)
Copiar a `reference/iamodel/`:

| Fichero | Para qué |
|---|---|
| `experiments/pose_extractor_clean.py` | Referencia de reglas de extracción y layout de salida |
| `experiments/custom_tracker.yaml` | Configuración de tracker de entrenamiento |
| `experiments/config.py` | Rutas, `CLIP_SCALE_HEIGHT`, modelo YOLO por defecto |
| `experiments/NPY_POSE_VISUALIZER_FRONTEND.md` | Especificación del formato `.npy` |
| `experiments/visualize_npy.py` | Visualizar un `.npy` para comprobaciones manuales |
| `experiments/yolo26s-pose.pt` | Pesos (alternativa: descargar con Ultralytics y registrar SHA-256) |
| `experiments/training/campaign/SAMPLE_production_chunk_mergedfps.npy` | Chunk real de producción (`[60, 8, 2]`) para tests de export |
| `docs/C1_POSE_PIPELINE_HANDOFF.md` | Descripción del pipeline de C1 |
| `docs/C1_POSE_PIPELINE_RESPUESTA_P0_P1.md` | Respuestas de C1 (BGR, tracker, umbrales, sesiones, empaquetado) |
| `docs/KANVIS_POSE_SHARED_PROJECT_SPEC.md` | Este documento |

Opcionales (contexto): `experiments/training/campaign/KANVIS_EXPERT_RESPONSE_20261014.md` y `KANVIS_CONTEXT_AND_RESEARCH_BRIEF.md` (análisis del experto y del contexto de negocio).

**Clips golden (el vídeo es lo importante; ver §10.2):** elegir 3–5 carpetas de `OUTPUT_BASE/data_result/{categoria}/{clip_name}/` (contienen `clip.mp4`, `meta.json`, `user_*/poses_full.npy`, `valid_mask.npy`, etc.). `OUTPUT_BASE` actual: `/media/8TB/DatosEntrenamiento/data_clips/data_yolo26s_061026`.
Copiar `clip.mp4` a `tests/golden/{clip}/clip.mp4` y el resto (`meta.json`, `user_*/`) a `tests/golden/{clip}/e1_iamodel/` como referencia E1. Los mismos `clip.mp4` hay que entregarlos a C1 para obtener E3.
Cubrir al menos: categoría 6 (robo), una categoría no robo, un clip con oclusión y uno con varias personas.

**No copiar:** `experiments/engine/build_triton_pose_plan.sh` y `export_yolo_pose_engine.py` (el primero genera un plan sin perfiles de batch, incompatible con el `config.pbtxt` de C1; el segundo produce un `.engine` de Ultralytics, que Triton no acepta), ni `training/train_model_operations.py` completo (el código relevante ya está en el Anexo A.5), ni los `.pt` de otras versiones de YOLO.

### D.2 Desde C1 (`kanvis-monitoring`, repo del equipo C1)
Pedir acceso de lectura o copia a `reference/c1/` de:

| Fichero | Para qué |
|---|---|
| `src/kanvis_monitoring/pose/triton/preprocess.py` | Letterbox y tensor (origen de la paridad y del hallazgo BGR) |
| `src/kanvis_monitoring/pose/triton/postprocess.py` | `decode_triton_pose_output`, `_decode_end2end_pose_output`, `MockYoloResult` |
| `src/kanvis_monitoring/pose/triton/client.py` | Cliente gRPC, FP16, SHM |
| `src/kanvis_monitoring/pose/guadia_export.py` | Contrato `.npy` de producción (`KEEP_KPS`, normalización, `valid_mask`) |
| `src/kanvis_monitoring/pose/detector.py` | `build_results_from_triton_batch`, `_apply_tracker_to_predict_result`, `_prepare_infer_image` |
| `src/kanvis_monitoring/pose/bytetrack_long.yaml` | Tracker de producción |
| `src/kanvis_monitoring/pose/guadia_custom_tracker.yaml` | Tracker alternativo (no activo) |
| `src/kanvis_monitoring/workers/continuous/track_sessions.py` | Semántica de sesiones |
| `scripts/export_triton_pose_onnx.py` | Export ONNX |
| `scripts/triton_pose_entrypoint.sh` | Compilación del plan (`trtexec`) |
| `scripts/triton_pose_config.py` | Plantilla de `config.pbtxt` |
| `scripts/check_triton_pose_batch.py` | Smoke test de batch |
| `scripts/offline_video_to_chunks.py`, `scripts/simulate_continuous_session.py` | Pipelines offline parciales |
| `docs/POSE_EXPORT_IAMODEL.md` | Perfil de export IAModel y runbook ONNX→plan |
| `config/monitoring.yaml`, `.env.docker.example` | Valores de configuración |

---

## Anexo E — Glosario
- **Bundle (`pose_bundle_id`)**: identificador versionado del conjunto modelo + preproceso + tracker + reglas de export, con hashes del ONNX, plan y `.pt`.
- **`poses_full`**: todas las filas temporales de una persona; `NaN` donde no hay medición válida.
- **`valid_mask`**: máscara por frame; `True` si hay ≥ 5 keypoints visibles.
- **end2end**: el modelo devuelve las detecciones ya filtradas (top-300) sin NMS externo.
- **Hold-last**: repetir la última pose cuando falta inferencia en un frame (`is_synthetic` en C1).
- **Plan TRT**: serializado TensorRT crudo que Triton carga como `model.plan`; no es el `.engine` de Ultralytics.
