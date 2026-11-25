# Window campaign — runbook operativo (entrenar y validar después)

Enfoque: **una corrida larga en GPU**, luego **una pasada de comparación**, sin ir ensayando ensemble a ensemble durante el entrenamiento.

## Qué tiene sentido (y qué no)

| Sí | No |
|----|-----|
| Dejar corriendo `run_overnight_batch.py` horas/día | Parar cada 2 épocas a probar pares 48+49 a mano |
| Al terminar: `batch_summary.csv` + elegir top por test/val | Confiar solo en test clip-window para “arreglé ses2” |
| Después del batch: **una** sesión replay (ses2, probes) sobre los 2–4 mejores checkpoints | Re-entrenar todo porque un ensemble no gustó |

El plan técnico sigue recomendando **no usar ses2 para elegir umbral**; ses2 es **aceptación** al final. Eso encaja con “entreno batch → luego valido”.

## Pipeline recomendado (1 noche)

```bash
cd IAModel/experiments/training/window_campaign
mkdir -p artifacts/logs

# Corrida completa stage2 (48,49,53,57) — ajusta --stage
nohup python3 run_overnight_batch.py \
  --data-root /ruta/data_result \
  --stage stage2 \
  > artifacts/logs/overnight_$(date +%Y%m%d).log 2>&1 &
```

Orden interno del batch:

1. `build_core_index.py`
2. `preflight_window_campaign.py --write-plan`
3. `train_window_campaign.py --stage <pilot|stage2|stage3>` (todos los IDs de esa fase, en serie)
4. `summarize_window_runs.py` → `artifacts/models/win_bin_full_baseline/batch_summary.csv`

**Duración orientativa:** pilot (48+49) @ 25 épocas ≈ unas horas según GPU y tamaño de `data_result`; stage2 (+53,57) ≈ el doble.

## Al volver: validación en 30 minutos

1. Abrir `batch_summary.csv` — ordenar por `macro_f1` / `test_f1_pos` / `binary_recall`.
2. Coger **top 2–4** `modelo_XX.pt` (no hace falta entrenar de nuevo).
3. **Replay offline** (misma máquina, sin reentrenar):
   - `campaign/score_pose_ensemble.py` con los NPys de ses2 / probes del brief.
   - Probar pares fijos: (48,49), (49,53), mejores dos del CSV.
4. Solo el ganador en replay pasa a despliegue / umbral.

## Fases de arquitectura (no “solo 2 modelos”)

| `--stage` | IDs | Cuándo |
|-----------|-----|--------|
| `pilot` | 48, 49 | Una noche si quieres ir conservador |
| `stage2` | 48, 49, 53, 57 | **Recomendado** para una noche “completa” con variedad |
| `stage3` | + 5, 11, 13 | Segunda noche si stage2 ya pasó checks básicos |
| `full` | pilot → stage2 → stage3 | Fin de semana largo |

Si **57** o **53** ganan en el CSV **y** en replay, se despliegan; no estás limitado a 48|49.

## Pendiente (no bloquea el batch en clips)

- **B8** sesiones `merged/` en el trainer (mejora FP real; Stage 2 datos).
- **B9** `replay_eval.py` automatizado post-batch (hoy: `score_pose_ensemble.py` a mano con top checkpoints).

## Comandos útiles

```bash
# Solo resumen si ya entrenaste
python3 summarize_window_runs.py

# Un stage sin batch
python3 train_window_campaign.py --data-root ... --stage pilot

# Ver qué se ejecutaría
python3 run_overnight_batch.py --data-root ... --stage stage2 --dry-run
```