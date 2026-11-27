# Comandos — window_campaign2

Directorio en el servidor:

```bash
cd ~/IAModel/experiments/training/window_campaign2
export DATA=/media/8TB/DatosEntrenamiento/datos_limpios_triton/data_result
```

## Fase 0 (solo medición, sin entrenar)

```bash
python3 build_core_index2.py --data-root "$DATA"

# Con checkpoints clip entero (campaña desplegada)
export CLIP_CKPT=~/IAModel/experiments/training/campaign/artifacts/runs/campaign_20261103_230735/models/bin_full
python3 phase0/run_phase0.py --data-root "$DATA" --split test \
  --full-clip-checkpoint-dir "$CLIP_CKPT"

cat artifacts/reports/phase0_report.md
```

Checkpoints ventana v1 (opcional, ya referenciados en config):

`../window_campaign/artifacts/models/win_bin_full_baseline/`

## Entrenamiento

```bash
# Smoke (200 clips, split nuevo)
python3 train_campaign2.py --data-root "$DATA" --stage pilot \
  --debug-clips 200 --force-new-split --experiment-id 49

# Benchmark completo (49+53+11), respeta manifiesto congelado
python3 train_campaign2.py --data-root "$DATA" --stage benchmark --resume

# Batch: P0 + train + eval
nohup python3 run_overnight_batch2.py --data-root "$DATA" \
  --full-clip-dir "$CLIP_CKPT" > artifacts/logs/overnight2.log 2>&1 &
tail -f artifacts/logs/overnight2.log
```

Progreso: `artifacts/logs/train_progress.json`

## Validación

```bash
python3 eval_campaign2_ensemble.py --data-root "$DATA" --pairs 49,53 11,49

python3 eval_campaign2_events.py --data-root "$DATA" --exp-id 49 --threshold 0.5
```

## Qué enviar tras una corrida

1. `artifacts/reports/phase0_report.md` (si corriste P0)
2. Log: `artifacts/logs/overnight2.log` o salida de `train_campaign2.py`
3. Líneas TEST de cada `modelo_XX.pt` (rec/FP del print final)
4. Salida de `eval_campaign2_ensemble.py` y `eval_campaign2_events.py`
5. `campaign2_config.json` si lo cambiaste en el servidor

## Notas

- `min_clip_seconds` = **0** en win2 (clips cortos entran; ver P0-7).
- Gate **sin** `protect_label_states` (simétrico).
- Positivo solo si overlap core ≥ **80%**; resto de ventanas en clip de robo → `ignore`.
- No uses `score_pose_ensemble.py` con checkpoints win2.