# Window campaign — comandos (solo binario)

Raíz: `IAModel/experiments/training/window_campaign`  
Sustituye `/ruta/data_result` por tu `data_result` real.

**Arquitecturas en corrida completa (`benchmark`): 6 modelos**

| exp | arquitectura |
|-----|----------------|
| 48, 49 | `tcn_gru` (prod) |
| 53 | `conformer_lite` |
| 57 | `ms_tcn` |
| 13 | `res_tcn` |
| 11 | `dilated_tcn` |

Multiclase **no** se usa. Celda actual: `win_bin_full_baseline` (full + weighted BCE). Variantes `hardened` / `filtered` son ablaciones futuras.

---

## a) PREFLIGHT — vista general + tiempos

```bash
cd IAModel/experiments/training/window_campaign

# Informe en consola (clips, chunks, core/ignore, augment, 6 arq, GPU/CPU)
python3 preflight_window_campaign.py \
  --data-root /ruta/data_result \
  --write-plan

# Mismo informe con solo 20 clips (probar sin leer todo el corpus)
python3 preflight_window_campaign.py \
  --data-root /ruta/data_result \
  --debug-clips 20
```

Salida: tablas en terminal + `artifacts/plans/win_bin_full_baseline/window_training_plan.json`

---

## b) ENTRENAMIENTO

### Debug rápido (N clips → pocos chunks, 2 épocas, 1 modelo)

```bash
python3 train_window_campaign.py \
  --data-root /ruta/data_result \
  --debug-clips 30 \
  --debug-epochs 2 \
  --experiment-id 48
```

### Corrida completa — 6 arquitecturas, 25 épocas (dejar corriendo)

```bash
# En primer plano
python3 train_window_campaign.py \
  --data-root /ruta/data_result \
  --stage benchmark

# Batch: core_index + preflight + 6 modelos + CSV (recomendado de noche)
nohup python3 run_overnight_batch.py \
  --data-root /ruta/data_result \
  --stage benchmark \
  > artifacts/logs/overnight_benchmark.log 2>&1 &
```

Checkpoints: `artifacts/models/win_bin_full_baseline/modelo_XX.pt`

---

## c) VALIDACIÓN (al terminar, sin reentrenar)

```bash
# Tabla comparativa test/val por modelo
python3 summarize_window_runs.py

# Abre:
# artifacts/models/win_bin_full_baseline/batch_summary.csv
```

Replay ses2 / ensembles (manual, con los 2–4 mejores del CSV):

```bash
cd ../campaign
python3 score_pose_ensemble.py \
  --poses-full /ruta/ses2/chunk_XXX/poses_full.npy \
  --model-a ../window_campaign/artifacts/models/win_bin_full_baseline/modelo_48.pt \
  --model-b ../window_campaign/artifacts/models/win_bin_full_baseline/modelo_49.pt
```

---

## Resumen una línea

| Paso | Comando clave |
|------|----------------|
| a) | `preflight_window_campaign.py --data-root ... --write-plan` |
| b) | `run_overnight_batch.py --data-root ... --stage benchmark` |
| c) | `summarize_window_runs.py` + `score_pose_ensemble.py` en top models |