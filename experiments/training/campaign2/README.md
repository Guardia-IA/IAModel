# Campaign2 (Training v2)

Pipeline **nuevo e independiente** de `../campaign/`. No modifica el entrenamiento legacy.

**Plan maestro:** [`IAModel/docs/TRAINING2_CAMPAIGN2_MASTER_PLAN.md`](../../../docs/TRAINING2_CAMPAIGN2_MASTER_PLAN.md)

## Sprint 3 — P1 train_v2 (implementado, MVP)

Entrenamiento con **ventanas 8s/12Hz** sobre la caché mmap, modelos:

- `masked_tcn64` (baseline)
- `gru64_pooled`

```bash
# Prueba rápida (2 épocas, mismo RUN_ID con caché):
./run_campaign2.sh p1-smoke --data-root /ruta/data_result

# Campaña P1 (2 jobs × 3 semillas = 6 entrenamientos por defecto):
./run_campaign2.sh preflight --p1 --skip-cache --data-root /ruta/data_result
./run_campaign2.sh train
# checkpoints: artifacts/runs/<RUN_ID>/jobs/v2_tcn64_baseline_s42/best.pt
```

Limitaciones MVP (mejora en sprints siguientes): etiqueta clip-level (MIL fino), norm isotrópica avanzada, split agrupado staged/real, eval P1 dedicada post-train.

## Sprint 2 — P0 baseline (implementado)

Evalúa los incumbent **#41** y **#48** en el split `val_dev` del manifest Campaign2:

| Modo | Descripción |
|------|-------------|
| **P0.A** (`legacy_clip`) | Mismo loader que `evaluate_campaign` (clip entero + `temporal_resize`) |
| **P0.B** (`windows`) | Ventanas 8 s @ 12 Hz, stride 1 s, agregación **max** por clip (caché mmap) |

Para cada modo: `single_user_only` **false** y **true**. Ensemble **AND @ 0.80** se calcula en `p0-summary` (umbrales congelados en config).

### Configurar checkpoints

En `campaign2_config.json` → `p0_incumbent.models[].path`, o variables de entorno:

```bash
export CAMPAIGN2_MODEL_41=/ruta/campaign/.../modelo_41.pt
export CAMPAIGN2_MODEL_48=/ruta/campaign/.../modelo_48.pt
```

### Flujo P0

```bash
cd IAModel/experiments/training/campaign2
./run_campaign2.sh new-run
./run_campaign2.sh probe
./run_campaign2.sh preflight --p0 --data-root /ruta/a/data_result
./run_campaign2.sh train
./run_campaign2.sh p0-summary
# o todo junto (con rutas a .pt):
./run_campaign2.sh p0-pipeline --data-root /ruta/a/data_result
```

Artefactos: `artifacts/runs/<RUN_ID>/jobs/p0*/`, `reports/p0_baseline.json`.

## Sprint 1 — Infra

| Comando | ¿Entrena Kanvis v2? |
|---------|---------------------|
| `preflight` | No — inventario + caché 12 Hz |
| `train` (sin `--p0`) | No — cola vacía o smoke |
| `preflight --p0` + `train` | No entrena — solo **eval** incumbent |

## Uso rápido (solo caché)

```bash
./run_campaign2.sh new-run
./run_campaign2.sh probe
./run_campaign2.sh preflight --data-root /ruta/a/data_result
./run_campaign2.sh summary
```

## Dependencias

- Mismo entorno que el legacy (`torch`, `numpy`).
- Lectura de clips vía **`collect_examples`** (solo lectura).
