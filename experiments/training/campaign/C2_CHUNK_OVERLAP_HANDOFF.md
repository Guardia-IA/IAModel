# Handoff C2 — ventana solapada entre chunks (`chunk_{k-1}_{k}`)

Documento para el **agente C2** (`c2_inference_service`): detectar robos cuyo gesto cae en la **frontera** entre dos chunks consecutivos de 8 s.

**Relacionado:** ensemble stage-1 → `C2_DEPLOY_ENSEMBLE_48_49_BIN_FULL.md` (u homólogo 49+53).

---

## Resumen en una frase

A partir de **`chunk_002`**, tras inferir el chunk actual, **concatenar** la **2ª mitad** del `poses_full` del chunk anterior con la **1ª mitad** del actual, inferir con el **mismo pipeline** que un chunk normal, y publicar en **`p1.json`** una ventana adicional **`chunk_{k-1}_{k}`** (ej. `chunk_001_002`) **inmediatamente después** del meta del chunk `k`.

---

## Problema (evidencia offline IAModel)

Con **modelo_48 + modelo_49**, MEAN @ **0,50**, `poses_full` + política de máscara alineada a prod:

| Ventana | p48 (aprox.) | p49 (aprox.) | MEAN | Alarma @0,50 |
|---------|--------------|---------------|------|----------------|
| `chunk_001` aislado | ~1,5 % | ~87 % | ~44 % | **No** |
| `chunk_002` aislado | ~0 % | ~0,4 % | ~0,2 % | **No** |
| **2ª mitad ch1 + 1ª mitad ch2** | ~100 % | ~100 % | ~100 % | **Sí** |

El robo en sesión `09e7c4e01e8845a48775aea09e6a132a` queda **partido** entre el final de `chunk_001` y el inicio de `chunk_002`. Sin solape, stage-1 **pierde el evento**.

Referencia de reproducción: carpeta *Robos 19 noviembre*, script `score_pose_ensemble.py`, modelos en `…/Kanvis/modelos_v2/modelo_48.pt` y `modelo_49.pt`.

---

## Comportamiento por orden de chunks

| Evento | C2 |
|--------|-----|
| Llega **`chunk_001`** | Inferencia normal. Guardar en **estado de sesión** `poses_full` (+ `valid_mask` si aplica). **No** emitir solape. |
| Llega **`chunk_002`** | 1) Inferencia `chunk_002`. 2) Construir solape con buffer de `chunk_001`. 3) Inferir solape. 4) En `p1.json`, **justo después** del bloque de `chunk_002`, añadir **`chunk_001_002`**. 5) Guardar `chunk_002` en buffer. |
| Llega **`chunk_003`** | `chunk_003` + **`chunk_002_003`** tras meta de `chunk_003`. |
| General | Para cada `chunk_k` con **k ≥ 2**, siempre **`chunk_{k-1}_{k}`** tras el chunk actual. |

Orden lógico en `p1.json`:

```text
… meta / ventana chunk_002 …
… meta / ventana chunk_001_002 …   ← solape, inmediatamente después
… meta chunk_003 …
… meta chunk_002_003 …
```

Ajustar nombres al esquema vigente (`c2_continuous_v2_p1_1.0`): puede ser `chunk_id: "chunk_001_002"` o campo análogo; debe ser **estable** y contener **los mismos campos de score** que una ventana motion normal.

---

## Construcción del npy de solape

**Por chunk (C1 / MinIO):** `poses_full.npy` → `[96, 8, 2]` (~8 s @ 12 Hz), opcional `valid_mask.npy` → `[96]`.

**Solape `chunk_{a}_{b}`** (`a` = k−1, `b` = k):

```python
poses_a = load_poses_full(session, chunk_a)   # desde buffer sesión o MinIO
poses_b = load_poses_full(session, chunk_b)
overlap = np.concatenate([poses_a[48:96], poses_b[0:48]], axis=0)  # shape [96, 8, 2]
```

- Cortes **48:96** y **0:48** sobre la **grilla densa** de 96 filas de `poses_full`, **no** sobre poses ya filtradas por `valid_mask`.
- **No** concatenar dos chunks enteros (192 filas) en un solo forward salvo diseño distinto acordado (comprime mal el gesto en `temporal_resize`).

### Preprocess e inferencia (idéntico al chunk único)

1. NaN → 0 (si aplica).
2. `normalize_sequence` → `add_velocity` → `temporal_resize(seq_len=96)`.
3. Forward **modelo_48** y **modelo_49** (cada uno con su `build_model` / engine).
4. `p_robo = softmax(logits)[1]`.
5. `robbery_score = (p48 + p49) / 2`; alarma si `≥ 0.50` (regla **mean**).

Reutilizar la **misma función** de scoring que ya usa C2 para un chunk; el solape es solo **otra entrada** `[96, 8, 2]`.

### Política `valid_mask` en solape (v1)

**Recomendación IAModel (alineada con prueba que detectó el robo):**

- Sobre el array `overlap`, **no** filtrar filas con máscara **antes** del preprocess estándar (equivalente a `score_pose_ensemble.py --no-valid-mask-filter`).
- Solo el tratamiento NaN→0 del pipeline habitual.

**Opcional v2:** `valid_overlap = concat(vm_a[48:96], vm_b[0:48])` y filtrar antes de preprocess; documentar en `p1` (`used_valid_mask_filter: true/false`). No usar en v1 sin re-validar paridad con IAModel.

---

## Ensemble (sin cambios respecto al deploy actual)

| Campo | Valor |
|-------|--------|
| Modelos | `modelo_48.pt` + `modelo_49.pt` (celda `bin_full`, campaña acordada) |
| Regla | `mean` |
| Umbral | `0.50` |
| Flags | `motion_theft_candidate` / `robbery_alarm` con la misma semántica que ventana normal |

Ver `C2_DEPLOY_ENSEMBLE_48_49_BIN_FULL.md` para carga, Triton y `p1.json` base.

### Campos extra recomendados en el bloque solape

```json
{
  "window_kind": "chunk_overlap",
  "chunk_id": "chunk_001_002",
  "overlap_chunks": ["chunk_001", "chunk_002"],
  "overlap_slices": { "prev": [48, 96], "next": [0, 48] },
  "T_raw": 96,
  "poses_source": "poses_full_concat",
  "ensemble_rule": "mean",
  "ensemble_threshold": 0.50,
  "p1": { "...": "modelo_48" },
  "p2": { "...": "modelo_49" },
  "robbery_score": 0.0
}
```

---

## Estado en C2

- Por **`session_id`**: buffer del último `poses_full` (+ `valid_mask` opcional) y `chunk_id`.
- Tras procesar `chunk_k`, actualizar buffer → datos de `chunk_k`.
- **`chunk_001`**: sin solape; no existe `chunk_000_001`.
- Si falta `chunk_{k-1}` (gap, OOO, reintento): **no** emitir `chunk_{k-1}_{k}`; log `overlap_skipped_reason`.
- **Orden:** emitir solape solo si `chunk_seq` es consecutivo (definir política si llegan fuera de orden).

---

## Downstream (C3)

- El solape es **otra ventana stage-1**; puede alarmar aunque los dos chunks aislados no lo hagan.
- C3 debe distinguir `window_kind: chunk_overlap` y evitar doble conteo sin reglas (fusión/dedup explícita).

---

## Criterios de aceptación

1. Sesión `09e7c4e01e8845a48775aea09e6a132a`: tras `chunk_002`, existe **`chunk_001_002`** con scores **~alineados** a IAModel (robo en frontera → mean alta, alarma @0,50 en el caso de prueba).
2. **Paridad** con:

```bash
cd IAModel/experiments/training/campaign

# Generar overlap (una vez; o desde C2 exportando el mismo array)
python3 <<'PY'
import numpy as np
from pathlib import Path
SESSION = Path("/ruta/sesión/09e7c4e01e8845a48775aea09e6a132a")
p1 = np.load(SESSION/"chunk_001/poses_full.npy")
p2 = np.load(SESSION/"chunk_002/poses_full.npy")
out = Path("/tmp/overlap_half_half.npy")
np.save(out, np.concatenate([p1[48:96], p2[0:48]]))
print(out)
PY

python3 score_pose_ensemble.py \
  --poses-full /tmp/overlap_half_half.npy \
  --no-valid-mask-filter \
  --model-a /ruta/modelo_48.pt \
  --model-b /ruta/modelo_49.pt \
  --rule mean --threshold 0.50 --json
```

3. Comparar `p_robo` y `robbery_score` C2 vs JSON anterior (tolerancia acordada, ej. 1e-4).
4. `chunk_001` solo: sin entrada de solape.
5. README / env C2 documenta overlap v1.

---

## Fuera de alcance (v1)

- Ventanas de 1 s + 1 s u otras longitudes (solo **mitad + mitad** = 96 filas).
- Cambio de modelos, umbral o `seq_len`.
- Modificar C1 (sigue exportando chunks de 8 s; el solape es **solo en C2**).

---

## Referencias

| Recurso | Uso |
|---------|-----|
| `score_pose_ensemble.py` | Preprocess + MEAN offline |
| `C2_DEPLOY_ENSEMBLE_48_49_BIN_FULL.md` | Modelos y umbral |
| `train_model_operations.py` | `normalize_sequence`, `add_velocity`, `temporal_resize` |

---

## Entregable pedido

PR C2 + ejemplo `p1.json` con `chunk_002` seguido de `chunk_001_002`, log de paridad con `score_pose_ensemble.py`, y nota para C3 sobre `window_kind`.

---

*IAModel — handoff solape chunks, nov 2026.*