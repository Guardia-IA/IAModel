# Campaign2 (Training v2 — Sprint 1)

Pipeline **nuevo e independiente** de `../campaign/`. No modifica el entrenamiento legacy.

**Plan maestro:** [`IAModel/docs/TRAINING2_CAMPAIGN2_MASTER_PLAN.md`](../../../docs/TRAINING2_CAMPAIGN2_MASTER_PLAN.md)

## ¿Qué hace cada comando? (importante)

| Comando | ¿Entrena Kanvis? |
|---------|------------------|
| `preflight` | **No.** Escanea clips, construye caché 12 Hz, escribe informe. Puede tardar minutos con ~19k clips. |
| `train` | **No** (Sprint 1). Solo ejecuta la cola JSON; por defecto **vacía**. |
| `train` tras `preflight --smoke-test` | **No.** Prueba GPU/scheduler (mini Conv1d de 8 pasos). |
| Entrenamiento real | **Sprint 2+** (`train_v2`, jobs P0/P1). |

## Sprint 1 (implementado)

| Componente | Rol |
|------------|-----|
| `campaign2_config.json` | Config raíz, paralelismo, temporal 8s/12Hz |
| `preflight_campaign2.py` | Inventario clips + **caché mmap 12 Hz** (`/dev/shm` si hay RAM) |
| `gpu_probe_campaign2.py` | VRAM probe → `max_parallel_jobs` |
| `gpu_scheduler.py` + `job_worker.py` | Cola multiproceso GPU |
| `train_campaign2_parallel.py` | Ejecuta `jobs/queue.json` |
| `run_campaign2.sh` | Orquestador shell |

## Uso rápido

```bash
cd IAModel/experiments/training/campaign2
./run_campaign2.sh new-run
./run_campaign2.sh probe
./run_campaign2.sh preflight --data-root /ruta/a/data_result
./run_campaign2.sh train
# o todo smoke (sin datos): ./run_campaign2.sh pipeline --skip-cache
```

Artefactos: `artifacts/runs/<RUN_ID>/` (caché, informes, logs por job).

## Dependencias

- Mismo entorno que el legacy (`torch`, `numpy`).
- Lectura de clips vía **`collect_examples`** en `train_model_operations.py` (solo lectura; no altera configs legacy).
