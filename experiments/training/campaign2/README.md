# Campaign2 (Training v2)

Pipeline **nuevo e independiente** de `../campaign/`. No modifica el entrenamiento legacy.

**Plan maestro:** [`IAModel/docs/TRAINING2_CAMPAIGN2_MASTER_PLAN.md`](../../../docs/TRAINING2_CAMPAIGN2_MASTER_PLAN.md)

## Raíz de datos (`--data-root`)

Puedes pasar:

- Una carpeta `.../data_result` (o `data_results`) directamente, **o**
- Una **carpeta principal** que contenga hijos `datos1/data_result`, `datos2/data_result`, etc.

El preflight lista todas las raíces descubiertas y fusiona clips en una sola caché.

## Pipeline debug (~minutos, 50 clips)

Valida caché + train paralelo + eval sin esperar horas:

```bash
./run_campaign2.sh debug-pipeline --data-root /ruta/principal/
# o solo: preflight --debug --p1 --data-root ...
```

Ajusta `debug` en `campaign2_config.json` (clips, épocas, 2 modelos). Para producción: **sin `--debug`**, sin `--max-clips`.

## Sprint 3 — P1 train_v2 + eval

```bash
./run_campaign2.sh new-run && ./run_campaign2.sh probe
./run_campaign2.sh preflight --p1 --data-root /ruta/principal/
./run_campaign2.sh train
./run_campaign2.sh p1-summary
./run_campaign2.sh p1-eval          # eval ventanas sobre cada best.pt
./run_campaign2.sh p1-eval-summary
watch -n 5 ./run_campaign2.sh p1-status   # durante train
```

P1 v2: TCN mean+max readout, cabeza concealment aux, MIL bag LSE, tail_positive, fps audit, hard neg manifest, split por sesión/prefix, calibración T, recall@FP, promote + debug-gate.
Pendiente largo plazo: intervalos anotados, loss MIL completa por evento, embargo temporal fino, despliegue prod 12Hz.

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
