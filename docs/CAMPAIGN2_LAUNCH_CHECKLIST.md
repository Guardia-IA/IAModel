# Campaign2 — checklist de lanzamiento (entrenamiento v2)

Objetivo: **dejar la pipeline implementada**, validar con **~10 clips** y lanzar el entrenamiento **completo (~19k clips)** para obtener **nuevos checkpoints v2** y métricas en la **nueva evaluación** (ventanas 8 s @ 12 Hz, recall@FP, calibración T, ensemble por semillas).

**No es objetivo de este lanzamiento:** shadow mode en producción, replay de sesiones completas, ni tomar #41/#48 como modelo a desplegar. Esos modelos pueden usarse **solo como referencia opcional** si activas P0 legacy.

---

## Configuración por defecto (consenso experto, mañana)

| Parámetro | Valor |
|-----------|--------|
| Ventana | 8 s @ 12 Hz (`seq_len=96`) |
| Arquitecturas | `masked_tcn64`, `gru64_pooled` |
| Semillas | 17, 42, 103 (3 por arquitectura) |
| Pose | `poses_full` (`campaign_matrix.pose_sources: ["full"]`) |
| Épocas prod | 100 (ver `time_estimate.p1_job_matrix`) |
| P0 legacy (#41/#48) | **desactivado** (`run_p0_legacy: false`) |

Tras el run: `p1-calibrate` → `p1-eval` en `val_dev` → `p1-seed-ensemble` → `ensemble_leaderboard.json`.

---

## En la máquina GPU (wudebian)

```bash
cd IAModel/experiments/training/campaign2
git pull   # traer fixes (imports ensemble, config 10 clips, etc.)

./run_campaign2.sh validate
```

Ruta típica de datos (ajusta si cambia):

```text
/media/8TB/DatosEntrenamiento/.../data_clips/
```

(o el `data_result` que uses en legacy; `--data-root` debe apuntar donde `collect_examples` encuentra clips).

---

## Paso 1 — Debug (~10 clips, ~minutos)

Comprueba que **toda la cadena** termina sin error: preflight matriz → train → calibrate → eval → seed-ensemble → leaderboard → debug-gate.

```bash
./run_campaign2.sh launch-debug --data-root /media/8TB/.../data_clips/
```

Equivalente: `./run_campaign2.sh debug-matrix --data-root ...`

**Esperado en debug:** métricas **no representativas** (pocos robos en val); lo importante es **exit code 0** y artefactos bajo `artifacts/runs/<RUN_ID>/`.

Revisar:

- `reports/p1_eval_summary.json` (o resumen en consola)
- `reports/p1_seed_ensemble.json`
- `reports/ensemble_leaderboard.json`
- `jobs/` sin jobs fallidos

---

## Paso 2 — Entrenamiento completo (~19k clips)

Cuando el debug pase:

```bash
# Opcional: estimación de tiempos
./run_campaign2.sh estimate-time --data-root /media/8TB/.../data_clips/ --matrix

./run_campaign2.sh launch-train --data-root /media/8TB/.../data_clips/
```

Dejar correr (preflight caché + **6 trains** P1: 2 arq × 3 seeds × 1 pose). Paralelismo según probe (5090).

Monitor:

```bash
export RUN_ID=$(cat artifacts/current_run_id.txt)   # o el ID impreso al new-run
./run_campaign2.sh p1-status --watch --run-id "$RUN_ID"
```

Al terminar, el **criterio de éxito** es el leaderboard v2 y métricas en splits congelados — **comparación cualitativa** con lo que ya tienes en prod, no re-despliegue de #41+#48.

```bash
./run_campaign2.sh p1-promote --run-id "$RUN_ID"
```

---

## Opcional — comparar #41/#48 en el mismo run

Solo si quieres filas P0 en el leaderboard (no necesario para entrenar v2):

1. En `campaign2_config.json`: `"run_p0_legacy": true`
2. Exportar rutas: `CAMPAIGN2_MODEL_41`, `CAMPAIGN2_MODEL_48`, etc.
3. Preflight con `--p0` o matrix con P0 activado

---

## Después del lanzamiento (no bloquea mañana)

- Ablation `filtered` vs `full` (duplica trains)
- Augment truncation, eval stride 4
- Replay sesiones / negativos reales / etiquetas por intervalo
- Shadow en producción

---

## Troubleshooting rápido

| Síntoma | Acción |
|---------|--------|
| `ModuleNotFoundError: evaluate_campaign` | `git pull`; `p1_seed_ensemble.py` / `ensemble_leaderboard.py` deben añadir `campaign/` al path |
| Preflight muy lento en debug | `--debug` limita inventario; `max_clips=10` en config |
| Split sin robos en val (debug) | Normal con N=10; `debug-gate` relaja mínimos |
| OOM en train | Bajar `max_parallel_jobs` o batch en config / probe |
