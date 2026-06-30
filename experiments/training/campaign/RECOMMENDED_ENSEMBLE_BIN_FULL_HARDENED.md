# Configuración recomendada — ensemble binario robo (post-campaña val)

Documento de handoff para implementar una **prueba de inferencia/evaluación** (test split u operación piloto).

**Origen:** campaña `experiments/training/campaign/` — resultados analizados en split **val** (no tunear en test; una sola pasada en test cuando la prueba esté lista).

**Fecha de referencia:** campaña ejecutada con `./run_campaign.sh all-bg`.

---

## Resumen ejecutivo

| Campo | Valor |
|-------|--------|
| **Modo** | **Binario** (robo vs resto) — **NO multiclase** |
| **Celda de entrenamiento** | `bin_full_hardened` |
| **Pose en inferencia** | **`full`** → `poses_full.npy` (+ `valid_mask.npy` si existe) |
| **Modelos** | **2 checkpoints** (ensemble, no uno solo) |
| **Regla de decisión** | **MEAN** de P(robo): `(P06 + P14) / 2 ≥ 0.68` → alarma |
| **Métricas val (referencia)** | F1 ≈ 73,3% · Recall ≈ 61,8% · FP ≈ 1,94% (~19 alarmas/día @ 1000 interacciones) |

---

## Qué NO usar

- Modelos de celdas `mc_*` (multiclase).
- Modelos de `bin_filtered`, `bin_full` (sin `_hardened`), etc.
- Un solo modelo con argmax (p. ej. solo `modelo_06.pt` → ~6,6% FP en val).
- `poses.npy` (filtered) si producción usa `poses_full.npy`.
- Regla **AND** (`P06≥0,68 y P14≥0,68`) — la config ganadora es **MEAN**, no AND.

---

## Celda `bin_full_hardened` — qué significa

| Parámetro | Valor |
|-----------|--------|
| `task` | `binary` — clase positiva = **categoría 6** (robo) del dataset original |
| `pose_source` | `full` |
| `class_map_id` | `full` (sin remap de clases; en binario solo importa exclude/remap si hubiera) |
| `aug_profile` | `fp_hardened` — más augment en negativos, loss `asymmetric`, `hard_negative_mining` |
| `experiment_ids` entrenados | `[6, 12, 13, 14, 17, 49]` (solo se usan **06** y **14** en este ensemble) |

Plan y augment de la celda:

```
experiments/training/campaign/artifacts/plans/bin_full_hardened/training_plan.json
experiments/training/campaign/artifacts/plans/bin_full_hardened/config_category_augmentation.json
```

En máquina de entrenamiento (Angel), rutas absolutas equivalentes:

```
/home/angel/IAModel/experiments/training/campaign/artifacts/plans/bin_full_hardened/training_plan.json
/home/angel/IAModel/experiments/training/campaign/artifacts/plans/bin_full_hardened/config_category_augmentation.json
```

---

## Checkpoints a cargar (los 2 obligatorios)

### Rutas relativas al repo

```
experiments/training/campaign/artifacts/models/bin_full_hardened/modelo_06.pt
experiments/training/campaign/artifacts/models/bin_full_hardened/modelo_14.pt
```

### Rutas absolutas (servidor entrenamiento)

```
/home/angel/IAModel/experiments/training/campaign/artifacts/models/bin_full_hardened/modelo_06.pt
/home/angel/IAModel/experiments/training/campaign/artifacts/models/bin_full_hardened/modelo_14.pt
```

Cada `.pt` es un checkpoint PyTorch de `train_model_operations.run_experiment` con:

- `task`: `"binary"`
- `positive_class`: `6`
- `num_classes`: `2` (índice 0 = no robo, índice 1 = robo)
- `model_state_dict`, `label_to_idx`, `config` (arquitectura), `seq_len`, `input_dim`

---

## Arquitecturas (según `model_config.py` EXPERIMENTS, índice 1-based)

### Experimento 06 → `modelo_06.pt`

| Campo | Valor |
|-------|--------|
| `arch` | `tcn` |
| `hidden_dim` | 256 |
| `dropout` | 0.3 |
| **`seq_len`** | **96** |
| `batch_size` | 32 (solo entrenamiento) |
| `lr` | 3e-4 |
| `epochs` | 60 |

### Experimento 14 → `modelo_14.pt`

| Campo | Valor |
|-------|--------|
| `arch` | `res_tcn` |
| `hidden_dim` | 256 |
| `num_blocks` | 3 |
| `dropout` | 0.3 |
| **`seq_len`** | **64** |
| `batch_size` | 32 |
| `lr` | 3e-4 |
| `epochs` | 40 |

**Importante:** cada modelo usa **su propio `seq_len`** del checkpoint. No compartir tensor de entrada entre ambos sin redimensionar temporalmente según cada `seq_len`.

---

## Regla de decisión (implementación)

Para cada clip / interacción / UID:

1. Cargar pose desde `data_result/{cat}/{clip}/.../poses_full.npy` (misma pipeline que entrenamiento: normalización, velocity, `temporal_resize` a `seq_len` del modelo).
2. Forward **modelo_06** → softmax → **P06** = probabilidad clase **1** (robo).
3. Forward **modelo_14** → softmax → **P14** = probabilidad clase **1** (robo).
4. **P_mean = (P06 + P14) / 2.0**
5. **Alarma robo si `P_mean >= 0.68`**

Pseudocódigo:

```python
p06 = prob_robo(modelo_06, pose, seq_len=96)  # del checkpoint
p14 = prob_robo(modelo_14, pose, seq_len=64)  # del checkpoint
p_mean = (p06 + p14) / 2.0
alarma = p_mean >= 0.68
```

Referencia en campaña: fila `ensemble_mean`, modelos `modelo_06|modelo_14`, thresholds `0.680|0.680` en:

```
artifacts/reports/bin_full_hardened/val_ensemble_grid.csv
artifacts/reports/_master/campaign_ensemble_val.csv
```

---

## Datos y split

| Concepto | Detalle |
|----------|---------|
| `data_root` | Definido en `training_plan.json` (`data_root`) o `GUADIA_DATA_RESULT_ROOT` / `model_config.DATA_RESULT_ROOT` |
| Split | UIDs en `training_plan.json` → `split_uids.test` para evaluación final; **val** ya usado para elegir umbral 0.68 |
| Filtros | Los del plan: `single_user_only`, `min_clip_seconds`, `min_valid_frames`, etc. |
| Sin augment | En eval/inferencia no aplicar data augmentation |

---

## Código existente reutilizable

| Módulo | Uso |
|--------|-----|
| `train_model_operations.build_model` | Construir red desde `checkpoint["config"]` |
| `train_model_operations.collect_examples` | Recolectar ejemplos con `pose_source="full"` |
| `post_training_binary_tools._forward_prob_pos` | Forward P(robo) por ejemplo (binario) |
| `evaluate_validation.py` | Eval batch; hay que **añadir** lógica ensemble MEAN o script nuevo |
| `test_model.py` | Filtros producción (ventana temporal, ROI, etc.) — capa opcional encima del ensemble |

---

## Petición para el agente (implementar mañana)

Implementar **script de prueba** (nombre sugerido: `campaign/test_ensemble_bin_full_hardened.py`) que:

1. Cargue **obligatoriamente** los dos checkpoints de `bin_full_hardened` (rutas arriba).
2. Evalúe en split **`test`** usando `training_plan.json` de la misma celda (sin mezclar train/val).
3. Aplique regla **MEAN( P06, P14 ) ≥ 0.68**.
4. Imprima y guarde CSV: `recall`, `precision`, `f1`, `fp_rate`, matriz de confusión binaria, conteo de alarmas.
5. Opcional: exportar falsos positivos (uid, `clip.mp4`, P06, P14, P_mean) a CSV + symlinks.

**Script ya disponible para listar FP en val (ensemble exacto):**

```bash
cd experiments/training/campaign
python export_ensemble_fp.py --split val
python export_ensemble_fp.py --split val --export-videos
# o: ./run_campaign.sh export-ensemble-fp
```

Genera:
- `artifacts/reports/bin_full_hardened/val_ensemble_fp_mean_modelo_06+modelo_14_t0.68.csv`
- `artifacts/reports/bin_full_hardened/val_ensemble_fp_summary.json`
- opcional: symlinks en `artifacts/fp_clips/bin_full_hardened/ensemble_mean_t0.68/`

**No** volver a barrer umbral en test; umbral **0.68** fijado en val.

---

## Métricas de referencia (solo val — no re-tunear)

Config: `bin_full_hardened`, ensemble_mean, `modelo_06|modelo_14`, umbral 0.68.

| Métrica | Val |
|---------|-----|
| F1 (positivo robo) | ~73,3% |
| Recall robo | ~61,8% |
| FP rate | ~1,94% |
| Alarmas/día (1000 interacciones) | ~19 |

Comparativa argmax mismo celda (no usar en producción):

| Modelo | F1 | Recall | FP |
|--------|-----|--------|-----|
| `modelo_06.pt` argmax | ~77,9% | ~78,5% | ~6,56% |

---

## Artefactos de campaña relacionados

```
experiments/training/campaign/
├── campaign_config.json
├── RECOMMENDED_ENSEMBLE_BIN_FULL_HARDENED.md   ← este fichero
└── artifacts/
    ├── models/bin_full_hardened/modelo_06.pt
    ├── models/bin_full_hardened/modelo_14.pt
    ├── plans/bin_full_hardened/training_plan.json
    └── reports/bin_full_hardened/
        ├── val_leaderboard.csv
        └── val_ensemble_grid.csv
```

Análisis completo previo: carpeta descargada `evaluacionkanvis/artifacts/` (leaderboard + ensemble binario).

---

## Checklist rápido antes de la prueba

- [ ] Existen `modelo_06.pt` y `modelo_14.pt` en `artifacts/models/bin_full_hardened/`
- [ ] Inferencia con **`poses_full.npy`**
- [ ] Checkpoints con `task=binary`, `positive_class=6`
- [ ] `seq_len` 96 (06) y 64 (14) leídos de cada checkpoint
- [ ] Regla: **media** ≥ 0.68, no argmax ni AND
- [ ] Evaluar en **test**, no en val
