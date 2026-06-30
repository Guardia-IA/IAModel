# Runs de mejora (sin machacar la campaña base)

Los artefactos de la campaña original viven en `artifacts/models/`, `artifacts/plans/`, etc.

Cada **run de mejora** usa `--run-id` y escribe en:

```
artifacts/runs/<run-id>/
  plans/<cell_id>/
  models/<cell_id>/
  reports/<cell_id>/
  reports/_master/
  run_meta.json
  preflight_summary.json
```

## Perfiles disponibles (`campaign_config_improve.json`)

| Perfil | Qué hace |
|--------|----------|
| `fp_hardened_hn` | fp_hardened + hard negatives (CSV FP) + boost augment en cats confusas |
| `fp_hardened_uniform4` | fp_hardened + augment uniforme ×4 (experimento A/B) |
| `fp_hardened_hn_uniform2` | Combinación moderada HN + uniform ×2 |

## Flujo recomendado

### 1. Exportar FP del ensemble baseline

```bash
cd experiments/training/campaign
python export_ensemble_fp.py --split val --export-videos
# → artifacts/reports/bin_full_hardened/val_ensemble_fp.csv (o similar)
```

### 2. Preflight + train en carpeta nueva

```bash
export RUN_ID=hn_v1
export HN_CSV=artifacts/reports/bin_full_hardened/val_ensemble_fp.csv
export PROFILE=fp_hardened_hn

./run_improve.sh preflight
./run_improve.sh train --exp-ids 6 14   # piloto: solo modelos del ensemble
./run_improve.sh eval
./run_improve.sh summary
```

O con argumentos explícitos:

```bash
python preflight_campaign.py --write-all \\
  --config campaign_config_improve.json \\
  --run-id hn_v1 \\
  --hard-negative-csv artifacts/reports/bin_full_hardened/val_ensemble_fp.csv \\
  --improve-profile fp_hardened_hn

python train_campaign.py --all --resume \\
  --config campaign_config_improve.json --run-id hn_v1

python evaluate_campaign.py --all \\
  --config campaign_config_improve.json --run-id hn_v1

python summarize_campaign.py \\
  --config campaign_config_improve.json --run-id hn_v1
```

### 3. Experimento augment uniforme (comparar vs baseline)

```bash
RUN_ID=uniform4_ab UNIFORM_OPS=4 PROFILE=fp_hardened_uniform4 ./run_improve.sh all-bg
```

## Qué activa cada palanca

| Palanca | CLI | Efecto |
|---------|-----|--------|
| Hard negatives UID | `--hard-negative-csv` | `hard_negative_uids.json` + sampler ×3 en train |
| Boost cats confusas | `--fp-category-boost 1.5` | Más variantes augment en categorías del CSV |
| Uniform ×N | `--uniform-ops-per-clip 4` | Mismo N augment por categoría (robo capped ~85%) |
| Perfil compuesto | `--improve-profile` | Combina train_opts (asymmetric loss, HN mining, ratio) |

## Comparar con baseline

- Baseline campaña: `artifacts/reports/bin_full_hardened/val_leaderboard.csv`
- Run mejora: `artifacts/runs/<run-id>/reports/bin_full_hardened/val_leaderboard.csv`
- Resumen run: `artifacts/runs/<run-id>/reports/_master/campaign_leaderboard_val.csv`

La campaña base **no se modifica** salvo que omitas `--run-id`.
