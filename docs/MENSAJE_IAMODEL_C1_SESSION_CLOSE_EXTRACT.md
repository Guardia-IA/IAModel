# Respuesta IAModel — cierre sesión C1 (`--session-close`, grid vídeo, `subject_track.json`)

Copiar/adaptar para Kanvis C1.

---

**Asunto:** Cierre sesión C1 — `--session-close` y alineación frame-a-frame (entrega 1)

Hola,

Recibido el contrato de cierre: **N sesiones / N person_id**, MP4 = **plano entero**, **no** enviáis `--export-track-id` (ID ByteTrack offline ≠ live). El listener invoca **`extract_clips.py`** directamente (el flag está en el subcomando `extract`, no solo en `run_extract.sh`).

---

## Entrega 1 (esta): `--session-close`

### Invocación acordada

```bash
export TRITON_EXTRACT_DECODE_BACKEND=opencv
python3 extract_clips.py extract \
  --video clip.mp4 \
  --out <work> \
  --cat gold \
  --parallel 1 \
  --validate \
  --session-close
```

(`run_extract.sh` es wrapper; mismo argv.)

### Comportamiento

| Flag / default | Cierre sesión | Batch entrenamiento (sin flag) |
|----------------|---------------|--------------------------------|
| `--session-close` | **Sí** | **No** |
| `single_user_only` en motor | **false** (solo con flag) | false si `export_primary_user_on_multi_track` (default JSON) |
| Gate `BODY_VISIBLE_*` + 20 % clip | **Relajado** al exportar | **Activo** → log `N track(s) filtros` si ningún track pasa |
| `export_primary_user_on_multi_track` | **Sí** → **un** `user_*` (más frames válidos) | Igual |
| `--no-single-user-only` | **No usar** (apaga primary) | — |
| `--export-track-id` | **No usáis** en piloto | — |

### Grid temporal (contrato esqueleto ↔ vídeo)

En motor `c1_like`, **`frame_policy=keep_all`** (default en `extract_stack`):

- **`poses_full.npy`**: shape `(T, 8, 2)` con **`T == frame_count`** del MP4 decodificado (ffprobe / filas decodificadas).
- **Fila `i`** = frame **`i`** del vídeo.
- Sin detección del track en ese frame → fila **NaN**; **`valid_mask.npy`** coherente (false donde no hay pose usable).
- **No** exportamos arrays “desde el primer frame del track” más cortos que el vídeo: si `T != decoded_T`, ese `user_*` no se escribe (error en log).

Esto es el mismo grid que gold N2 / `compare_c1_golden` (índice i↔i). `--session-close` **no cambia** inferencia ni `keep_all`; solo **permite escribir** `user_*` cuando los filtros de entrenamiento habrían dejado 0 carpetas.

Tras export, **`prune_to_primary_user`**: un solo `user_{track_id}`; `meta.json` incluye `iamodel_primary_user_track_id` si hubo poda.

Metadatos:

- `iamodel_session_close_export: true`
- `users[].export_relaxed: true` si se exportó relajando BODY_VISIBLE
- `iamodel_export_track_id: null` (sin `--export-track-id`)

---

## Entrega 2: `--subject-bbox-json`

Flag en **`extract_clips.py extract`**:

```bash
python3 extract_clips.py extract \
  ... \
  --session-close \
  --subject-bbox-json /path/subject_track.json
```

- Si la ruta **no existe**, el CLI termina con error (evita caída silenciosa del cierre).
- Misma semántica que **`ClipExtractor(subject_track_json=...)`** en kanvis-triton-pose: `load_subject_track` (`bbox_by_frame` o `frames[]`), **`pick_detection_for_session`** con IoU mínimo **0,3** ( `session_close_min_iou` en `ClipExtractConfig` ).
- Export: **`user_0`** ( `session_close_user_id` ), grid denso **T = frames decodificados**, sin poda primary.
- `meta.json`: `iamodel_subject_bbox_json`, `frame_policy: session_subject_iou`.

Sin `--subject-bbox-json`, `--session-close` sigue usando **primary user** + relajación BODY_VISIBLE (entrega 1).

Listener: activar `KANVIS_IAMODEL_EXTRACT_SUBJECT_JSON=1` cuando despleguéis rama con este flag.

---

## Comprobaciones (sesión ref. `ff3c1f32200a4ed6b8f9f84808ccebe9`)

En wudebian (Triton up, código con `--session-close` desplegado):

```bash
cd ~/IAModel/experiments/triton_pose_extract
SESSION=/home/angel/TestKanvis/ff3c1f32200a4ed6b8f9f84808ccebe9   # ajustar ruta clip
OUT=/tmp/session_close_ff3c1f
CLIP="$SESSION/clip.mp4"   # o clip_buffer.mp4 según scratch

# A) sin session-close → debe seguir sin user_* (repro filtered=2)
python3 extract_clips.py extract --video "$CLIP" --out "$OUT/batch_like" \
  --cat gold --parallel 1 --validate -v 2>&1 | tee /tmp/extract_no_sc.log

# B) con session-close → 1 user_*, T alineado
python3 extract_clips.py extract --video "$CLIP" --out "$OUT/close" \
  --cat gold --parallel 1 --validate --session-close -v 2>&1 | tee /tmp/extract_sc.log

CLIP_ROOT="$OUT/close/gold/$(basename "${CLIP%.mp4}")"
python3 scripts/verify_session_close_output.py "$CLIP_ROOT"
cat "$CLIP_ROOT/meta.json"
python3 -c "
import numpy as np, json
from pathlib import Path
r=Path('$CLIP_ROOT')
m=json.loads((r/'meta.json').read_text())
u=next(r.glob('user_*'))
pf=np.load(u/'poses_full.npy'); vm=np.load(u/'valid_mask.npy')
print('shape', pf.shape, 'valid_mask.sum()', int(vm.sum()), 'frame_count', m['frame_count'])
"
```

**Resultados numéricos** (`meta.json`, shape, `valid_mask.sum()`) los pegamos en el hilo en cuanto ejecutemos B sobre el scratch si aún existe (no reproducible desde el repo IAModel sin acceso al clip).

Criterios PASS (script + `--validate`):

1. Exactamente **un** `user_*/poses_full.npy` y `valid_mask.npy`.
2. `poses_full.shape[0] == meta.frame_count == valid_mask.shape[0]`.
3. `valid_mask.sum() > 0`.
4. `iamodel_session_close_export == true`; `export_relaxed: true` en el usuario si aplicó relajación.
5. Sin `--session-close`, mismo clip → **0** `user_*` (comportamiento batch intacto).

---

Saludos,  
IAModel / triton_pose_extract
