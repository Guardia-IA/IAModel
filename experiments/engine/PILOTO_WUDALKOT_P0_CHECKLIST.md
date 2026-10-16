# Piloto wudalkot — checklist P0 (plan TRT → Triton READY → IAModel → chunk)

Orden acordado con C1. Ejecutar en **wudalkot** (GPU + Docker). Ajustar rutas del repo kanvis-monitoring si difieren.

## 0. Preflight

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

## 1. Artefactos YOLO en host

```bash
YOLO_DIR=/home/angel/kanvis/piloto_v1/yolo
ls -la "$YOLO_DIR"/yolo26s-pose.pt "$YOLO_DIR"/yolo26s-pose.onnx
```

Si falta `.pt`:

```bash
cd ~/IAModel/experiments/engine
./download_yolo_pose_pt.sh "$YOLO_DIR" yolo26s-pose.pt
```

> **Nota (15-oct-2026):** según `docs/C1_POSE_PIPELINE_RESPUESTA_P0_P1.md`, el sidecar `triton-pose` de C1
> ya exporta el ONNX (dynamic, batch 32) y compila `model.plan` por su cuenta (`triton_pose_entrypoint.sh`).
> Usa la **Opción A** de abajo; la Opción B (script IAModel) genera un plan sin perfiles de batch y solo vale para pruebas.

## 2. Compilar plan TRT crudo (no `.engine` Ultralytics)

**Opción A — runbook C1** (`docs/POSE_EXPORT_IAMODEL.md` / `triton_pose_entrypoint.sh` en kanvis-monitoring).

**Opción B — script IAModel** (ONNX ya generado: paso 1 del script puede reutilizar el existente):

```bash
cd ~/IAModel/experiments/engine
source ~/IAModel/experiments/venv/bin/activate   # ultralytics para ONNX si hace falta
./build_triton_pose_plan.sh \
  --host-dir /home/angel/kanvis/piloto_v1/yolo \
  --yolo yolo26s-pose.pt \
  --output-name yolo26s-pose.triton.plan
```

Verificar que **no** es formato Ultralytics (magic 833):

```bash
python3 - <<'PY'
import struct
p = "/home/angel/kanvis/piloto_v1/yolo/yolo26s-pose.triton.plan"
with open(p, "rb") as f:
    le = struct.unpack("<I", f.read(4))[0]
print("first u32 LE =", le, "(833 = Ultralytics .engine — MAL; ~1953657958 = plan TRT OK)")
PY
```

Opcional: `validate_triton_pose_plan.py` del sidecar C1 sobre el mismo fichero.

## 3. Publicar plan para Triton

- Copiar/symlink según compose C1: **`model.plan`** en el model repository del sidecar `triton-pose`.
- Variables (ver `.env.docker.example` kanvis-monitoring): apuntar a **`yolo26s-pose.triton.plan`**, no al `.engine` Ultralytics antiguo.

```bash
# Ejemplo genérico (adaptar a vuestro sync):
# cp "$YOLO_DIR/yolo26s-pose.triton.plan" /ruta/triton_model_repo/yolo_pose/1/model.plan
```

## 4. Redeploy sidecar + smoke

Desde el directorio del compose del piloto:

```bash
docker compose pull triton-pose   # si aplica
docker compose up -d triton-pose c1   # nombres según stack
docker compose ps
docker compose logs triton-pose --tail 80
```

Smoke (scripts C1 en repo kanvis-monitoring):

```bash
./scripts/verify_triton_pose_stack.sh
# o
python3 scripts/check_triton_pose_batch.py
```

Criterio: modelo **READY**, inferencia **1×3×640×640** OK, sidecar estable (sin restart loop).

## 5. Export IAModel en prod (MinIO)

En `.env.docker` del piloto (y redeploy C1 si lee env al arranque):

```bash
CONTINUOUS_POSE_EXPORT_IAMODEL_FULL=true
KANVIS_POSE_BUNDLE_ID=yolo26s-pose_v1   # si no está ya por default
```

Confirmar que **no** se monta `.engine` Ultralytics como `model.plan`.

## 6. Chunk de aceptación

1. Reproducir **vídeo corto conocido** en el piloto (misma cámara/flujo habitual).
2. Localizar en MinIO/S3 el folder del chunk (`chunk_*/`).
3. Comprobar ficheros y meta:

```bash
# En una copia local del chunk:
ls -la poses_full.npy valid_mask.npy keypoint_mask.npy pose_frame_timestamps.npy meta.json pose_bundle.json
python3 - <<'PY'
import json, numpy as np
from pathlib import Path
d = Path(".")  # carpeta del chunk
meta = json.loads((d / "meta.json").read_text())
for k in ("iamodel_full", "export_min_kp_conf", "export_min_visible_kps", "pose_bundle_id"):
    print(k, meta.get(k))
pf = np.load(d / "poses_full.npy")
vm = np.load(d / "valid_mask.npy")
print("poses_full", pf.shape, "valid_mask", vm.shape, "valid_frames", int(vm.sum()))
PY
```

Esperado: `iamodel_full=true`, conf **0.5**, visible **5**, `pose_bundle_id=yolo26s-pose_v1`, `pose_bundle.json` presente.

## 7. Entrega a IAModel

Enviar ruta S3/local del chunk + sesión `diag.json` si aplica → cruce con contrato train (`valid_mask`, NaN en `poses_full`).

---

**Fuera de este checklist:** P1 batch “como prod”; parche C2 `valid_mask` en motion (paralelo, no bloquea P0).
