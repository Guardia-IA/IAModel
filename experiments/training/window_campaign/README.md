# Window campaign (8 s chunks from existing `data_result` clips)

Bloque **independiente** de `campaign/`: no modifica planes ni entrenamiento histórico.  
Objetivo: entrenar con **ventanas de 96 frames (~8 s @ 12 Hz)** tomadas de los mismos `.npy` por acción que ya tenéis en `data_result`, con **augmentación sintética por ventana** (misma familia que `validate_npy` / rejilla determinista).

## Requisitos

- Misma estructura que el pipeline actual: `data_result/{categoria}/.../user_X/poses_full.npy` (+ opcional `valid_mask.npy`).
- Python + PyTorch (mismo entorno que `train_campaign.py`).

## Flujo operativo (entrenar de noche, validar al día siguiente)

Ver **`OPERATIONAL_RUNBOOK.md`**: `run_overnight_batch.py` encadena core_index → preflight → todos los experimentos de una fase → `batch_summary.csv`.

```bash
nohup python3 run_overnight_batch.py --data-root /ruta/data_result --stage stage2 \
  > artifacts/logs/overnight.log 2>&1 &
```

## Flujo manual

```bash
cd IAModel/experiments/training/window_campaign

# 1) Preflight: cuenta ventanas, escribe plan
python3 preflight_window_campaign.py \
  --data-root /ruta/a/data_result \
  --write-plan

# 2a) Entrenar en memoria (sin escribir N ventanas a disco)
python3 train_window_campaign.py \
  --data-root /ruta/a/data_result \
  --experiment-id 48 \
  --task binary

# 2b) Opcional: materializar ventanas a .npy (cache/revisión)
python3 materialize_windows.py \
  --data-root /ruta/a/data_result \
  --out-dir ./artifacts/window_cache/run001
```

Si no pasáis `--data-root`, se usa `config.get_data_result_roots()[0]` o `OUTPUT_BASE/data_result`.

## Configuración

`window_config.json`:

| Campo | Default | Descripción |
|-------|---------|-------------|
| `window_len` | 96 | Frames por ventana (8 s @ 12 Hz) |
| `stride` | 48 | Paso entre ventanas en el mismo clip |
| `pad_mode` | `repeat_last` | Si T &lt; 96: `repeat_last` o `zeros` |
| `mask_policy` | `keep_grid` | `keep_grid`: cortar en la grilla densa; no filtrar filas antes del corte |
| `synthetic_variants` | … | Reusa perfiles de augment de campaña (determinista + on-the-fly) **por ventana** |

## Troceo (todas las categorías, incluidas 6 y 14)

- **96 filas = 8 s @ 12 Hz** (no 9 s). Cortes en memoria desde cada `poses_full.npy`.
- **6 y 14 también se parten**: no hay excepción por “clip limpio”; si `T<96` → una ventana con padding; si `T≥96` → ventanas con stride 48 (solape 50%).
- Ejemplo **16 s (192 filas)** → **3 chunks**: `[0:96)`, `[48:144)`, `[96:192)`.

Detalle y augment: **`WINDOW_CAMPAIGN_TRIAL_PLAN.md` §6**.

## Augment sintético

Igual filosofía que la campaña anterior, pero la unidad es **ventana** (`clip_uid::w{start}`), no el clip entero:

- Antes: ~**5 vistas sintéticas por `.npy`** (clip entero). Ahora: ~**5 por cada chunk de 8 s** (mismo manifest del clip aplicado al recorte). Con 16 s → 3 chunks × 5 ≈ **15 filas** lógicas de train (sin contar augment aleatorio on-the-fly).
- Variantes deterministas desde `validate_npy` manifests (si existen en caché del clip).
- Política (ver `WINDOW_CAMPAIGN_TRIAL_PLAN.md` §6.7): augment en **todas** las ventanas de train, **incluidos solapes 50%** y categorías **6/14**; val/test sin augment aleatorio.
- Config: `extra_manifest_views_per_clip: 4` → **5** filas deterministas por ventana (identidad + 4).
- Augment aleatorio on-the-fly en train (`aug_profile` en config), adicional a las 5 fijas.

## Artefactos

```
window_campaign/
  artifacts/
    plans/          # window_training_plan.json
    models/         # checkpoints window_bin_full/
    window_cache/   # opcional, materialize_windows
```

## Relación con MOTION_FP_RETRAIN_PLAN.md

Este bloque implementa la **fase datos mínima v1** (slices 96 desde clips existentes).  
Overlap entre sesiones, preprocess v2 y hard negatives de prod siguen en el roadmap; se pueden añadir scripts aparte sin tocar `campaign/`.