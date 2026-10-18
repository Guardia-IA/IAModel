# Brief para agentes: Triton + `.npy` por clip

Mensaje de handoff para integradores (IAModel, batch offline, herramientas que consumen pose). Repo: **kanvis-triton-pose**.

Ver también: [INTEGRATION_CONSUMERS.md](INTEGRATION_CONSUMERS.md), [DOCKER_POSE_TOOLS.md](DOCKER_POSE_TOOLS.md), [C1_AGENT_BRIEF.md](C1_AGENT_BRIEF.md), [TRITON_SYSTEM_SHM.md](TRITON_SYSTEM_SHM.md), [TRITON_EXTRACT_PERF.md](./TRITON_EXTRACT_PERF.md) (rendimiento offline vs C1 / NVDEC).

---

## Qué es este servicio

No es un API REST de “sube vídeo → devuelve npy”. Es **NVIDIA Triton** con el modelo **`yolo26s-pose`** (TensorRT plan, salida YOLO **end2end**). El cliente debe:

1. Leer frames del clip (OpenCV).
2. **Preprocess** (letterbox 640, color, tensor).
3. **Inferir por gRPC** en batches.
4. **Decode** `(300, 57)` → cajas + 17 keypoints.
5. **Tracker** (ByteTrack por defecto) para asociar persona por `track_id`.
6. **Export** → escribir `poses_full.npy`, `valid_mask.npy`, etc. por `user_{track_id}/`.

Eso ya está implementado en **`pose_core`** (`ClipExtractor`, CLI `extract-clips`). **Reutiliza ese código**; no reimplementes el contrato a mano salvo que sepas replicar [`config/pose.yaml`](../config/pose.yaml) y el decode.

---

## Conectividad (piloto wudalkot / 5090)

| Contexto | URL gRPC | HTTP (health / ready) |
|----------|----------|------------------------|
| **Host / SSH** | `127.0.0.1:47811` o `kanvis-pilot.ai:47811` | `http://127.0.0.1:47810` |
| **Contenedor en la misma red Docker que Triton** | `triton-pose:8001` | `http://triton-pose:8000` |
| **C1 same-host vía gateway** | `kanvis-pilot.ai:47811` (+ `extra_hosts: host-gateway` si el DNS no resuelve dentro del contenedor) | `47810` en host |

- **Protocolo inferencia:** **gRPC** (`tritonclient[grpc]`), no HTTP para infer.
- **Modelo:** `yolo26s-pose` (comprobar: `GET /v2/models/yolo26s-pose/ready` → 200).
- **System SHM:** solo si cliente y Triton están en el **mismo bare metal** con `ipc: host` en ambos (`KANVIS_POSE_TRITON_SYSTEM_SHM_ENABLED=true`). Para **extracción offline por clips**, suele bastar **gRPC sin SHM** (`false`).

Smoke desde el bundle:

```bash
export PYTHONPATH="/ruta/kanvis-triton-pose:$PYTHONPATH"
python scripts/check_triton_pose_batch.py --url triton-pose:8001 --model yolo26s-pose --no-config
# o --url kanvis-pilot.ai:47811 desde host/C1 con gateway
```

---

## Contrato Triton (bajo nivel — solo la inferencia)

**Entrada (tensor `images`):**

- Forma: **`(B, 3, 640, 640)`** NCHW.
- Tipo: **FP16** (config `input_fp16: true`).
- Valores: **float en [0, 1]** (no 0–255).
- Preprocess alineado con [`config/pose.yaml`](../config/pose.yaml): redimensionar frame (p. ej. ancho máx. 640 / alto 360), **`input_color: bgr` → `model_color: rgb`**, **letterbox 640**, luego `to_tensor`. Ver `pose_core.preprocess`: `prepare_infer_image`, `letterbox`, `to_tensor`.

**Salida (tensor `output0`; el nombre puede sincronizarse desde metadata):**

- Forma típica: **`(B, 300, 57)`** — detecciones end2end (conf + bbox + 17×3 kp en coords letterbox).
- El cliente debe **`decode_end2end`** (`pose_core.postprocess`) con el `LetterboxMeta` del frame y umbral ~`parse_conf * 0.5` (p. ej. `parse_conf` 0.18 → decode ~0.09).

**Cliente Python de referencia:** `pose_core.triton_client.TritonPoseClient.infer(batch)` — espera NCHW float32 en [0,1], envía FP16 por gRPC, devuelve numpy de salida.

**Timeouts:** primera infer en GPU puede tardar **1–3 min**; usar `infer_timeout_seconds` generoso (p. ej. 180–300).

---

## Camino recomendado: un clip → carpeta con `.npy`

Usa la CLI del repo (no hables con Triton “a pelo” para el npy final):

```bash
# PYTHONPATH o contenedor pose-model-prepare / iamodel con vendor montado
python scripts/run_extract_clips.py extract-clips \
  --video /ruta/clip.mp4 \
  --out /ruta/salida \
  --triton-url triton-pose:8001 \
  --config config/pose.yaml \
  --bundle-json triton/model-repo/yolo26s-pose/1/pose_bundle.json \
  --tracker bytetrack \
  --batch 8
```

Lote de clips:

```bash
python scripts/run_extract_clips.py extract-clips \
  --video-dir /ruta/clips \
  --out /ruta/data_result \
  --triton-url kanvis-pilot.ai:47811 \
  --tracker bytetrack
```

### Estructura de salida (por clip)

```text
{out}/{nombre_clip}/
  clip.mp4
  meta.json              # fps, frame_count, pose_bundle_id, users[], yolo_backend=triton_tensorrt_plan
  pose_bundle.json       # manifiesto del modelo
  user_{track_id}/
    poses_full.npy       # (T, 8, 2) coords normalizadas 0–1, KP COCO 5–12; NaN si conf ≤ 0.5
    poses.npy            # solo frames “válidos” (≥5 KP visibles)
    keypoint_mask.npy    # (T, 8) bool
    valid_mask.npy       # (T,) bool por frame
```

- **T** = número de frames procesados del vídeo (política `keep_all` por defecto en `config/pose.yaml` → `clip.frame_policy`).
- Filtros: el track debe pasar ratios mínimos de cuerpo visible (`BODY_VISIBLE_MIN_FRAMES`, etc.) o no se escribe `user_*`.

Reglas export IAModel / MinIO: [POSE_EXPORT_IAMODEL.md](POSE_EXPORT_IAMODEL.md).

---

## Dependencias Python del cliente

`numpy`, `opencv-python-headless`, `tritonclient[grpc]`, `PyYAML`; tracker vía `pose_core.tracker` (mocks estilo Ultralytics).

---

## Errores frecuentes

| Síntoma | Causa probable |
|---------|----------------|
| `HTTP 000` / connection refused | Triton caído, puertos mal en `.env.docker`, URL incorrecta (`47811` host vs `8001` red interna) |
| `Modelo no READY` | Esperar compilación TRT (`model.plan`) o revisar `docker compose logs triton-pose` |
| Clip sin `user_*` | Con `single_user_only`: **multi-track esperado** (no elige dominante); si no, decode/tracker o filtros `BODY_VISIBLE_*`. Ver [TRITON_EXTRACT_C1_HANDOFF.md](./TRITON_EXTRACT_C1_HANDOFF.md) |
| Drift vs IAModel histórico | Distinto `pose_bundle_id`, `model_color`, tracker o umbrales |

---

## Identidad del modelo

Tras `pose-model-prepare`:

```bash
cat triton/model-repo/yolo26s-pose/1/pose_bundle.json
```

Incluir `pose_bundle_id` en `meta.json` de clips nuevos (p. ej. `yolo26s-pose_v1`).

---

## Resumen rápido

| Pregunta | Respuesta |
|----------|-----------|
| ¿IP/puerto? | gRPC **47811** (host) / **8001** (Docker); HTTP **47810** / **8000** |
| ¿Qué envío a Triton? | Batches **NCHW FP16 [0,1] letterbox 640 RGB**, nombre `images` |
| ¿Qué devuelve Triton? | **`output0`**: `(B, 300, 57)` raw end2end |
| ¿Cómo obtengo `.npy`? | **`extract-clips`** + `ClipExtractor` (preprocess → infer → decode → track → export) |
| ¿Smoke? | `check_triton_pose_batch.py --url … --model yolo26s-pose` |

Copia en C1: `vendor/kanvis-triton-pose` vía [`scripts/export_bundle_for_c1.sh`](../scripts/export_bundle_for_c1.sh).
