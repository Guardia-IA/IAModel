# Rendimiento: extracción offline Triton (IAModel) vs C1

**Contexto:** batch `extract-csv` en host con **RTX 5090**, Triton `yolo26s-pose`, `pose_core.ClipExtractor`.  
**Síntoma medido (Oct 2026):** `nvidia-smi` SM **~1–17 %** durante batch; mediana **~23,6 s/clip**; `clip_buffer` **28–46 s** pared para tramos de **4–18 s** de vídeo.

**Conclusión:** el cuello de botella **no** es TRT en GPU; es **decode + tracker + I/O**. C1 en producción es rápido porque usa **otro tubo de ingest** (GStreamer **NVDEC**, colas, preprocess paralelo). El offline actual **no** replica eso.

Referencias: [C1_POSE_PIPELINE_HANDOFF.md](./C1_POSE_PIPELINE_HANDOFF.md), [TRITON_EXTRACT_C1_HANDOFF.md](./TRITON_EXTRACT_C1_HANDOFF.md), [AGENT_TRITON_NPY_BRIEF.md](./AGENT_TRITON_NPY_BRIEF.md), [KANVIS_POSE_SHARED_PROJECT_SPEC.md](./KANVIS_POSE_SHARED_PROJECT_SPEC.md) §6.

---

## 1. Qué hace hoy IAModel (`triton_pose_extract`)

| Fase | Implementación | GPU |
|------|----------------|-----|
| Recorte CSV | FFmpeg `-c copy` (sin decode) | No |
| Duración / cola jobs | OpenCV `VideoCapture` | No |
| **Frames del clip** | **`pose_core` → OpenCV `VideoCapture` (CPU decode)** | No |
| Preprocess letterbox 640 | CPU (`pose_core.preprocess`) | No |
| Inferencia pose | **Triton gRPC TRT** | **Sí (picos)** |
| Tracker | ByteTrack + **`yolo26s-pose.pt` Ultralytics (CPU)** | No |
| Export npy | CPU + disco 8TB | No |

**NVDEC:** **no** se usa en este camino.  
**System SHM:** en IAModel estaba **desactivado** por defecto en el cliente (`use_system_shm=False`); ver §3.1 para activarlo en mismo host.

---

## 2. Qué hace C1 (rápido en RTSP)

| Fase | C1 (`kanvis-monitoring`) |
|------|---------------------------|
| Ingest | **GStreamer**, `enable_nvdec: true` |
| Resize | 640×360 antes de letterbox |
| Triton | Pool + batch entre cámaras, **preprocess workers** (8+) |
| Tracker | Mismo `pose_core` / decode TRT, pero alimentado con pipeline continuo |
| FPS | Submuestreo (~15 fps sesión); **offline IAModel = todos los frames** (`keep_all`) |

El camino caliente (**`pipeline.py`, `triton_pool.py`, SHM, workers preprocess**) **no vive** en `ClipExtractor` CSV; el spec dice que **permanece en C1** ([KANVIS_POSE_SHARED_PROJECT_SPEC.md](./KANVIS_POSE_SHARED_PROJECT_SPEC.md) D3).

**Paridad de npy** = mismo `pose_core` + `pose.yaml` + bundle. **Paridad de velocidad** = hace falta **paridad de ingest/decode** o reutilizar scripts offline de C1.

---

## 3. Plan de optimización (fases)

### Fase 0 — Ya en IAModel (sin parar filosofía Triton)

| Acción | Efecto esperado |
|--------|-----------------|
| `export_primary_user_on_multi_track: true` | Más OK, mismo coste/clip |
| `skip_existing` solo con `user_*/poses_full.npy` | Reanudar sin saltar fallos |
| FFmpeg `clip_scale_height: 0` | Recorte rápido |
| **`parallel_clips: 2`** en 5090 (no 4) | Menos contención CPU/disco cuando GPU idle |
| **`triton.use_system_shm: true`** + `ipc: host` cliente/Triton | Menos copia CPU en infer (mismo bare metal) |
| Triton **`gpu_instance_count`** / `max_batch_size` | Subir si métricas Triton muestran cola |

**Comando batch recomendado (5090, mismo host Triton):**

```bash
# triton_pose_extract_config.json:
#   "parallel_clips": 2,
#   "triton": { "use_system_shm": true, "gpu_instance_count": 2, "max_batch_size": 32 }
#   "triton_batch_size": 8  (o 16 si VRAM aguanta)

export KANVIS_POSE_TRITON_SYSTEM_SHM_ENABLED=true   # si pose_core lo lee del env
export KANVIS_TRITON_POSE_ROOT=~/kanvis/piloto_v1/kanvis-triton-pose

./run_extract.sh extract-csv --config triton_pose_extract_config.json \
  --input-root ... --output-base ... -v
```

**Validación:** durante 1 clip, `nvidia-smi dmon` debería subir **más** en ventanas de infer; si sigue &lt;20 %, el cuello sigue siendo decode/track.

---

### Fase 1 — Ajuste operativo (sin cambiar repos)

1. **Partir el batch por tipo de CSV**
   - **Cortos** (`clip_buffer`, tramos HMS) → un run con `--parallel 2`.
   - **`_full_` / action-rec** → otro run (o `--parallel 1`), expectativa **~60–80 s/clip**.
2. **Disco:** salida en **NVMe** local y rsync a 8TB (opcional) si el cuello es I/O.
3. **No mezclar** con YOLO engine en la misma GPU sin cuotas (Triton + Ultralytics compiten).

---

### ¿NVDEC para leer cada clip desde IAModel? (sin vendor)

**Pregunta típica:** “¿Podemos usar NVDEC aquí para procesar frames más rápido?”

| | |
|---|---|
| **A nivel hardware** | **Sí.** Cada MP4 H.264/H.265 se puede decodificar con **NVDEC** (GStreamer `nvh264dec` / `nvdec`, FFmpeg `-hwaccel cuda` + `h264_cuvid`, PyNvVideoCodec, etc.). |
| **Con el batch actual (`extract-csv` → `ClipExtractor`)** | **No, no con un flag.** Tras el recorte FFmpeg (`-c copy`), **`pose_core` vuelve a abrir el archivo y lee frame a frame con OpenCV `VideoCapture` (decode CPU)**. IAModel no entra en ese bucle. |
| **¿Basta NVDEC solo en el recorte FFmpeg?** | **Casi nunca.** El recorte **no decodifica** (stream copy). Aunque re-codificaras el segmento con decode GPU en `ffmpeg_cut`, **OpenCV volvería a decodificar el MP4 en CPU** al inferir — dos pasadas o una pasada CPU igual de cara. |
| **¿Trucos sin tocar `pose_core`?** | **Poco fiables:** `OPENCV_FFMPEG_CAPTURE_OPTIONS` / backend FFmpeg con hwaccel (depende del build de OpenCV; en muchos hosts sigue siendo CPU). No sustituye el tubo GStreamer de C1. |

**Qué sí sería “usar NVDEC aquí” de verdad:**

1. **Recomendado (Fase 2):** `FrameSource` / `decode.backend: gstreamer_nvdec` en **`kanvis-triton-pose`**, mismo contrato que monitoring — IAModel solo pasa la ruta del clip y la config.
2. **Spike IAModel (caro):** reimplementar el bucle de `ClipExtractor` en `experiments/triton_pose_extract` (NVDEC → preprocess → Triton → ByteTrack → export), duplicando lógica de vendor y obligando a seguir parches de `pose_core`.
3. **Puente C1:** invocar scripts offline de **`kanvis-monitoring`** si ya decodifican con NVDEC y escriben el mismo npy/chunks (evaluar contrato antes de producción).

**Resumen:** NVDEC **sí es la palanca correcta** para bajar los ~23–30 s/clip cuando el cuello es decode; **no es enchufable** al pipeline actual sin cambiar **quién produce los frames** (vendor o fork del extract).

---

### Fase 2 — Objetivo “como C1” (cambio en **`kanvis-triton-pose`** + acuerdo C1)

**Meta:** pipeline offline **MP4 archivo** con decode **NVDEC** y alimentación TRT similar a C1.

| Entregable | Repo | Descripción |
|------------|------|-------------|
| **`FrameSource` NVDEC** | `kanvis-triton-pose` | GStreamer `nvh264dec` / `nvdec` o FFmpeg `-hwaccel cuda` → frames BGR sin `VideoCapture` CPU |
| **Preprocess pool** | `pose_core` o wrapper IAModel | N workers CPU/GPU al estilo `triton_preprocess_workers: 8` |
| **Inferencia** | Triton existente | Batches más grandes, SHM, 2–4 instances en 5090 |
| **Tracker** | Evaluar GPU / lighter | C1 usa `.pt` CPU para ByteTrack; offline podría limitar resolución track |
| **Config** | `pose.yaml` | `decode: { backend: gstreamer_nvdec \| opencv, enable_nvdec: true }` |

**Scripts C1 a evaluar** (offline parcial, mismo contrato TRT):

- `kanvis-monitoring/scripts/simulate_continuous_session.py`
- `kanvis-monitoring/scripts/offline_video_to_chunks.py`

IAModel debería poder invocarlos o converger en un **`extract-clips-fast`** en vendor.

**Petición formal a C1** (copiar):

> Necesitamos offline batch MP4 con **mismo TRT/decode/tracker** que producción, pero **ingest NVDEC** (no OpenCV CPU). Medidas: SM GPU 1–17 %, 30 s pared en clip 10 s. ¿Roadmap: `FrameSource` nvdec en `ClipExtractor`, reutilizar módulo GStreamer de monitoring, o script offline oficial? ¿Valores `parallel_clips`, `triton_batch_size`, SHM e `instance_group` para 5090?

---

### Fase 3 — Arquitectura objetivo (diagrama)

```text
                    ┌─────────────────────────────────────┐
  MP4 (CSV)         │  IAModel extract-csv (futuro)        │
       ───────────► │  FFmpeg copy (segmento)            │
                    │       ↓                            │
                    │  GStreamer/NVDEC → BGR 640×360     │  ◄── como C1
                    │       ↓                            │
                    │  Preprocess pool (N workers)       │
                    │       ↓                            │
                    │  Triton TRT (SHM, batch 8–16)      │  ◄── GPU
                    │       ↓                            │
                    │  decode_end2end + ByteTrack        │
                    │       ↓                            │
                    │  export npy + primary_user (IAModel)│
                    └─────────────────────────────────────┘
```

Hoy falta el bloque **NVDEC + preprocess pool**; por eso la 5090 no se parece a C1.

---

## 4. ¿Parar el batch actual?

| Seguir | Parar y relanzar con Fase 0–1 |
|--------|-------------------------------|
| ~8k+ npy ya válidos | Tras Fase 2, mucho más rápido |
| No requiere código nuevo | Config + posible vendor en semanas |

**Recomendación:** terminar run actual si falta &lt;2 días **o** parar si prioridad es **optimizar antes** de quemar 70 h más. Lo exportado con `user_*/` **se conserva** (`skip_existing`).

---

## 5. Benchmark de aceptación (5090)

Tras cada cambio, medir **un** `clip_buffer` ~10 s:

```bash
/usr/bin/time -f 'wall %e s' ./run_extract.sh extract \
  --config triton_pose_extract_config.json \
  --video /ruta/clip_10s.mp4 --out /tmp/bench --cat 0 --parallel 1 -v
```

| Métrica | Baseline Oct 2026 | Objetivo Fase 0 | Objetivo Fase 2 (C1-like) |
|---------|-------------------|-----------------|---------------------------|
| Pared clip 10 s | ~30 s | ~15–20 s | **~5–10 s** |
| GPU SM media | ~1–17 % | ~25–40 % | **~50–80 %** en infer |
| Clips/h (4 workers) | ~200–600 | ~800–1200 | **2000+** (orientativo) |

---

## 6. YOLO engine local (histórico)

`pose_extractor_clean` + TensorRT/Ultralytics **en proceso** suele saturar GPU mejor porque **no** separa decode OpenCV + gRPC + tracker CPU. Útil para **volumen urgente**; **no** sustituye `pose_bundle_id` Triton en Campaign2 sin acuerdo.

---

## 7. Checklist rápido antes del próximo batch

- [ ] Triton READY + smoke batch  
- [ ] `parallel_clips: 2`, `use_system_shm: true` (mismo host)  
- [ ] `export_primary_user_on_multi_track: true`  
- [ ] Preflight con `--probe-videos` en todas las raíces  
- [ ] `nvidia-smi dmon` durante 5 min de batch  
- [ ] Ticket C1 / issue `kanvis-triton-pose`: **NVDEC offline FrameSource**  
- [ ] Documentar clips/h y SM en `logs/extract_csv_*.log` + este doc  

---

*IAModel — Oct 2026*
