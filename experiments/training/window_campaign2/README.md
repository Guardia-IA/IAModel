# window_campaign2

Native **8 s** buffers, **symmetric** quality gate, **v2 labels** (robbery fragments → `ignore`, never `neg`), checkpoint by **recall @ FP cap**.

Implements Phase 0 (measure) + Phase 1 (train) from `IAModel/docs/PROPOSAL_NEXT_STEP_WINDOW_VS_CLIP.md`.

Does **not** touch `window_campaign/` artifacts or deployed `modelo_48` / `modelo_49`.

## Layout

| Path | Role |
|------|------|
| `campaign2_config.json` | Cell `win2_bin_native_baseline` |
| `phase0/run_phase0.py` | P0 report → `artifacts/reports/phase0_report.md` |
| `train_campaign2.py` | Train exp 49, 53, 11 (default benchmark) |
| `eval_campaign2_ensemble.py` | Window-level test metrics |
| `eval_campaign2_events.py` | Event OR recall |
| `run_overnight_batch2.py` | P0 + train + eval |

Shared preprocessing/sampler/loss: reuses `../window_campaign/` modules.

## Quick start (server)

```bash
cd IAModel/experiments/training/window_campaign2
export DATA=/media/8TB/DatosEntrenamiento/datos_limpios_triton/data_result

# 1) Core index (copies from v1 if missing)
python3 build_core_index2.py --data-root "$DATA"

# 2) Phase 0 (optional full-clip dir for H1 matrix)
export CLIP_CKPT=/path/to/campaign_20261103_230735/models/bin_full
python3 phase0/run_phase0.py --data-root "$DATA" --full-clip-checkpoint-dir "$CLIP_CKPT"

# 3) First split manifest (once) or use frozen after
python3 train_campaign2.py --data-root "$DATA" --stage pilot --debug-clips 200 --force-new-split

# 4) Full benchmark
nohup python3 run_overnight_batch2.py --data-root "$DATA" --full-clip-dir "$CLIP_CKPT" \
  > artifacts/logs/overnight2.log 2>&1 &
```

See `COMANDOS_PASOS.md` for validation-only commands.

## Outputs

- Checkpoints: `artifacts/models/win2_bin_native_baseline/modelo_{49,53,11}.pt`
- Split: `artifacts/plans/win2_bin_native_baseline_split_manifest.json`
- Logs: `artifacts/logs/`