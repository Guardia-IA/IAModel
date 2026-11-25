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

## Reparto train / val / test y augment

**No es la campaña clásica** (`train_campaign.py` + `training_plan.json` fijado por celda), pero usa la **misma lógica**:

| Garantía | Cómo |
|----------|------|
| Split por **clip** (UID), no por ventana | `split_examples_stratified_by_uid` — un `.npy`/usuario no cruza splits |
| Estratificado por **categoría** | Val/test reparten robos y negativos por carpeta |
| Sin fugas | `assert_no_uid_leak_between_splits` + manifiesto `artifacts/plans/*_split_manifest.json` |
| **Val/test sin augment** | Sin `expand_manifest`, sin padding simulado en negativos, `augment_on_the_fly=False`, sin `forced_ops` (assert en runtime) |
| Solo **preprocess** en eval | `normalize_sequence` + velocidad (`keep_grid_v2`) — igual que inferencia, sin mirror/rotate/etc. |

Orden: **split clips → ventanas → calidad → core labels → (solo train) padding sim + manifest → datasets**.

**Anti-overfitting en validación**

- `train ∩ val = ∅`, `train ∩ test = ∅`, `val ∩ test = ∅` a nivel **UID de clip** y **ruta `.npy`** — si hay fuga, el entrenamiento **aborta** (`certify_disjoint_splits`).
- Tras la **primera** corrida se guarda `artifacts/plans/win_bin_full_baseline_split_manifest.json`. Con `use_frozen_manifest: true`, los re-entrenos **reutilizan** los mismos clips en val/test (nunca mezclados con train).
- Al final del batch: `verify_split_disjoint.py` vuelve a comprobar el manifiesto (+ datos si pasas `--data-root`).
- Para re-sortear todo: borra el manifiesto o `train_window_campaign.py ... --force-new-split`.

**Calidad de datos**

- **Clip entero:** `clip_filters.min_clip_seconds` (default **4 s**) — como antes en `collect_examples`.
- **Cada chunk:** `quality_gate` descarta ventanas con teletransporte (salto ≫ mediana), muchos frames vacíos, poca visibilidad, o contenido &lt; **2 s** (último trozo con padding: mínimo **1 s**).
- Ajustar en `window_config.json`; el preflight muestra cuántos chunks se tiran y por qué.

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