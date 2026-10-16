# Kanvis Training v2 / Campaign2 — Plan maestro (autocontenido)

**Fecha:** 2026-10-16  
**Idioma:** español (documento de implementación interna).  
**Alcance:** nuevo pipeline de entrenamiento y evaluación en `IAModel/experiments/training/campaign2/` (y módulos compartidos en `training/`), incorporando las mejoras acordadas tras `TRAINING_AUDIT.md` y `TRAINING_EXPERT_RESPONSE_20261015.md`, con **paralelización total** en GPU RTX 5090.  
**Fuera de alcance:** extracción de pose (C1), inferencia en producción (C2), SAM3/heurísticas (salvo diseño de handoff).

---

## 1. Contexto del proyecto

**Kanvis** detecta hurtos en supermercado a partir de **esqueletos** (YOLO26s-pose + tracker), no de la imagen completa del cliente. Stage 1 (este plan) clasifica secuencias de movimiento; Stage 2 (SAM3, HAR, C3) confirma o descarta alarmas.

**Datos actuales (campaña octubre, referencia):**

- ~19 043 clips etiquetados, **~2 592 clips cat. 6 (robo/ocultación)**, una tienda, **9 cámaras**.
- Etiquetado: una acción por clip (mp4 → npy + categoría humana); subclips con categoría propia.
- Producción: sesiones por persona/track, chunks ~8 s, ventanas solapadas; puede haber **cortes por oclusión o cambio de ID** (varias sesiones por la misma persona física).

**Objetivo de negocio (prioridad):**

- **FP primero:** preferencia explícita del owner: mejor **50 % recall con 0 FP/día** que 75 % con 5 FP/día.
- Meta operativa de trabajo: acercarse a **≤1 FP/día** con **~1 000 sesiones/día**, y **recall por evento** lo más alto posible dentro de ese presupuesto (Stage 2 ayuda después).

**Estado del modelo legacy (campaign v1):**

- Mejor ensemble val (clip-level): **#41 + #48, AND @ 0.80** → F1 ~76 %, recall ~64 %, **11 FP / 1 619 negativos (~0,68 %)**.
- Producción referencia: **~20 FP / 654 sesiones (~3,1 %)** — no comparable directamente con val por ventanas/sesión y población distinta.
- Problemas conocidos: bug de checkpoint (H1), split por clip, `single_user_only`, `temporal_resize`, umbral optimizado en val, etiqueta “robo” que incluye guardar el móvil, etc. (detalle en auditoría y respuesta experta).

**Por qué Campaign2:** implementar entrenamiento **alineado con producción**, evaluación **honesta**, y **usar toda la capacidad de la 5090** entrenando muchos jobs en paralelo en lugar de **un modelo tras otro** (como `train_campaign.py` hoy).

---

## 2. Principios de diseño

1. **No romper v1:** `campaign/` y `train_model_operations.py` siguen intactos; v2 vive en **`campaign2/`** + código nuevo (`train_v2/`, `dataset_v2.py`, etc.) reutilizando solo utilidades estables (paths, class maps donde aplique).
2. **Un contrato temporal único** (train = eval = producción objetivo): ventanas **8 s @ 12 Hz → 96 pasos**, stride **1 s**, máscaras, tiempo físico.
3. **Métrica de selección:** recall por **evento/sesión** con presupuesto de **FP/sesión o FP/día**, no `val_acc` ni F1 clip-level como criterio principal.
4. **Paralelización por defecto:** cada unidad de trabajo independiente (experimento × semilla × celda × fase de eval) es un **job**; el orquestador lanza hasta **`max_parallel_jobs`** concurrentes en la misma GPU si la VRAM lo permite.
5. **Reproducibilidad:** cada job tiene `job_id`, semilla, hash de preprocesado, manifest de split agrupado, logs y artefactos bajo `campaign2/artifacts/runs/<RUN_ID>/`.
6. **Fases incrementales:** cada fase tiene criterios de aceptación; no avanzar a arquitecturas nuevas sin baseline honesto (Fase 0–1).

---

## 3. Layout de carpetas propuesto

```
IAModel/experiments/training/
├── train_model_operations.py      # legacy (solo lectura / parches mínimos H1 si urgentes)
├── model_config.py                # legacy EXPERIMENTS
├── campaign/                      # legacy pipeline
├── train_v2/                      # NUEVO: dataset, normalización, modelos, loss, métricas
│   ├── __init__.py
│   ├── dataset_windows.py         # ventanas 8s/12Hz, MIL, máscaras
│   ├── normalize.py               # torso, isotrópico, stats
│   ├── models/                    # masked_tcn, gru_pooled, ...
│   ├── train_loop.py              # AdamW, checkpoint por recall@FP
│   └── metrics_session.py         # agregación sesión/evento
└── campaign2/
    ├── README.md                  # puntero a este doc + comandos rápidos
    ├── campaign2_config.json      # jobs, paralelismo, fases, data roots
    ├── class_maps/                # copia o symlink a campaign/class_maps
    ├── jobs/                      # definiciones JSON de experimentos v2 (no 60×8 celdas)
    ├── preflight_campaign2.py
    ├── train_campaign2_parallel.py   # orquestador multiproceso
    ├── evaluate_campaign2_parallel.py
    ├── gpu_scheduler.py           # cola, VRAM probe, límites
    ├── run_campaign2.sh
    └── artifacts/runs/<RUN_ID>/
        ├── jobs/                  # un subdir por job_id
        ├── plans/
        ├── models/
        ├── reports/
        └── logs/
```

Documentación de referencia (solo lectura):

- `IAModel/docs/TRAINING_AUDIT.md`
- `IAModel/docs/TRAINING_EXPERT_RESPONSE_20261015.md`
- `IAModel/docs/TRAINING2_CAMPAIGN2_MASTER_PLAN.md` (este archivo)

---

## 4. Paralelización total (RTX 5090)

### 4.1 Problema actual

- `train_campaign.py` ejecuta un **bucle secuencial** `for exp_id in exp_list: run_experiment(...)` en **un solo proceso**, **una GPU al 100 %** por modelo, ~**15–25 min/modelo** (real ~7× más que el preflight estimaba en octubre).
- Campaña completa: **8 celdas × 20 modelos ≈ 160 entrenamientos** → días de pared; en la práctica ~**2 días** para un subconjunto curado.
- La GPU **no se comparte** entre entrenamientos aunque modelos pequeños (TCN ~0,1–3 M params) **no llenen 32 GB**.

### 4.2 Objetivo Campaign2

- Tratar cada **(job_spec, seed)** como proceso independiente.
- Mantener una **cola de jobs** y un **scheduler** que arranque hasta **`N` procesos hijo** en paralelo (`N = max_parallel_jobs`), cada uno con:
  - `CUDA_VISIBLE_DEVICES=0` (una GPU física),
  - **`torch.cuda.set_per_process_memory_fraction`** o límite inferido por probe,
  - **semilla y `job_id` únicos**,
  - escritura en directorios **disjoint** (`artifacts/runs/<RUN_ID>/jobs/<job_id>/`).

**Regla:** paralelizar **entre modelos/semillas/evaluaciones**, no duplicar el mismo job. Dentro de un solo entrenamiento, subir **`num_workers` del DataLoader** (CPU) y batch size hasta saturar GPU **de ese proceso**; el scheduler baja `N` si hay OOM.

### 4.3 Estimación de capacidad (5090, 32 GB VRAM)

| Tipo de job | VRAM orientativa (batch típico) | Jobs paralelos conservadores | Agresivo (tras probe) |
|-------------|----------------------------------|------------------------------|------------------------|
| TCN v2 width 64, seq 96, batch 64 | ~2–4 GB | 4–6 | 8–12 |
| TCN-LSTM 256 (#41-like), batch 32 | ~6–10 GB | 2–3 | 4–5 |
| Evaluación (solo forward, batch grande) | ~1–2 GB por modelo | 8–15 | 20+ |
| Preflight / inventario (CPU) | 0 GPU | N_cpu = núcleos/2 | — |

**Meta operativa inicial:** **`max_parallel_jobs = 8`** tras script `gpu_probe_campaign2.py`; ajustar automáticamente a **`floor(0,85 × VRAM_libre / VRAM_por_job)`** con reserva **`vram_reserve_mb = 2048`**.

Si el probe demuestra que caben **10** TCN v2 pequeños → entrenar **10 semillas/configs a la vez** → tiempo de pared ≈ **`T_secuencial / 10`**.

### 4.4 Componentes del scheduler

| Componente | Responsabilidad |
|----------|-----------------|
| `gpu_scheduler.py` | Cola FIFO/prioridad, slots libres, reintentos OOM (bajar batch o serializar), registro JSONL `scheduler_events.jsonl` |
| `train_campaign2_parallel.py` | Expande `campaign2_config.json` → lista de jobs; lanza workers (`subprocess` o `ProcessPoolExecutor` con **spawn**) |
| `evaluate_campaign2_parallel.py` | Jobs de scoring en GPU: un worker por checkpoint o batch de checkpoints |
| `run_campaign2.sh` | Fases: `probe → preflight → train-parallel → eval-parallel → summarize` |
| Variables env | `CAMPAIGN2_MAX_PARALLEL`, `CAMPAIGN2_VRAM_FRACTION_PER_JOB`, `CAMPAIGN2_DATALOADER_WORKERS` |

**Política OOM:** (1) reintentar job con `batch_size // 2`; (2) si falla, marcar slot ocupado y reducir `max_parallel_jobs` global para el resto del run; (3) nunca corromper checkpoint parcial (escribir a `.tmp` y rename).

**CPU/RAM:** con 10 procesos × `num_workers=4` → 40 workers DataLoader; cap **`CAMPAIGN2_CPU_WORKERS_TOTAL`** (p.ej. 32) repartido entre jobs activos para no matar el host.

### 4.5 Qué paralelizar en cada fase

| Fase | Jobs paralelizables |
|------|---------------------|
| P0 Re-eval legacy | Cada checkpoint × split × `single_user` on/off × ventana (grid reducido) |
| P0 H1 retrain incumbent | #41/#48 × 3 seeds = 6 jobs |
| P1 Ablation preprocess | Factorial acotado (4–8 configs) × 3 seeds |
| P1 Matriz principal v2 | ~6 arquitecturas × 3 seeds = 18 jobs |
| P2 Filtros post (E12) | Cada filtro = eval batch sobre mismos logits (CPU/GPU mix, paralelo por sesión-shard) |
| Preflight | Shards por `data_root` / categoría (multiproceso CPU) |

**No paralelizar:** elección de umbral en el conjunto T (una sola pasada serial sobre scores agregados); escritura del manifest de split (una vez, determinista).

---

## 5. Fases de implementación y entrenamiento

Cada fase incluye **entregables de código**, **criterios de aceptación** y **plan de jobs paralelos**.

---

### Fase 0 — Corrección y baseline honesto (P0)

**Objetivo:** medir la verdad antes de entrenar “mejor”.

**Cambios de código (campaign2 + parche mínimo legacy opcional):**

1. **H1:** en v2, snapshot inmutable (`copy.deepcopy` o save directo al mejor epoch); prohibido `load_state_dict` de referencia viva.
2. **`evaluate_campaign2` modo replay:** ventanas 8 s / 12 Hz / stride 1 s; agregación **max** por sesión; métricas **FP/sesión** y recall evento.
3. **Comparativa población:** evaluar con **`single_user_only` true y false** (jobs paralelos 2×).
4. **Incumbent frozen:** cargar #41/#48 de `campaign_20261006_231053`, **sin re-optimizar umbral en test**; umbrales congelados exportados en JSON.

**Jobs paralelos ejemplo (Fase 0):**

```text
Job A1: incumbent #41, val, windows, single_user=true
Job A2: incumbent #41, val, windows, single_user=false
Job A3: incumbent #48, ...
Job A4: ensemble AND @0.80 frozen, ...
Job A5–A6: mismo en split test (solo si T congelado en val)
```

**Aceptación Fase 0:**

- Informe `reports/p0_baseline.json` con FP/sesión, recall evento, intervalos.
- Golden test: mismo npy → score idéntico legacy vs v2 loader (tolerancia float documentada) **o** diff explicado.

**Paralelismo:** hasta **6–8 eval jobs** simultáneos (forward ligero).

---

### Fase 1 — Contrato de datos, preprocesado y entrenamiento alineado (P1)

**Objetivo:** dejar de entrenar “clip deformado” y pasar a **ventanas físicas** con máscaras y etiquetas utilizables.

**Implementación (`train_v2/`):**

| Tema | Especificación |
|------|----------------|
| Tiempo | Resample **12 Hz** por índices de frame / fps en meta; ventana **8 s → 96**; stride **1 s** |
| Máscaras | No eliminar frames del tensor temporal; **mask** por joint/frame; NaN → 0 + flag |
| Normalización | Ancla caderas (fallback hombros); escala torso robusta; **(x×W/H, y)** isotrópico |
| Features | x,y,vx,vy, validez, distancias muñeca–torso (~62 canales); estandarización train-only |
| Labels | Clip cat.6 = **bolsa MIL**; subconjunto ~200 clips con intervalo `[t0,t1]`; crops sin gesto = unknown (sin loss theft) |
| Loss | **BCEWithLogits** theft + auxiliar concealment (peso 0,2); sin apilar focal+asymmetric+mining al inicio |
| Sampler | Por **grupo** (clip/event_id): 25 % pos / 25 % hard neg / 50 % neg; cap windows/clip |
| Optim | AdamW, cosine+warmup, grad clip; **100 epochs max**, early stop por **recall@FP** en val agrupado |
| Augment | flip, jitter temporal 0,9–1,1, joint dropout, gaps 0,1–0,5 s, crops aleatorios; **sin** shift/scale nulos |
| Split | **Grupos:** sesión/grabación/actor (componentes conexas de overlap); embargo temporal; val/test cronológico |

**Configuración `single_user_only`:**

- Default campaign2: **`false`** en train/eval, con **track objetivo** vía `user_cat` / manifest (no etiquetar todos los esqueletos del mp4 como positivos).

**Jobs paralelos Fase 1 (matriz reducida, no 160 runs):**

| job_id | Descripción | seeds |
|--------|-------------|-------|
| `v2_tcn64_baseline` | Masked TCN dilated width 64 (receta experta) | 17, 42, 103 |
| `v2_incumbent48_fixed` | #48 arquitectura legacy pero loader v2 + mismo seq 96 | 17, 42, 103 |
| `v2_gru64_pooled` | BiGRU + masked mean/max (fix pooling) | 17, 42, 103 |
| `ablation_no_mil` | MISMO que tcn64 sin MIL (diagnóstico) | 42 |
| `ablation_old_norm` | MISMO con norm legacy (diagnóstico) | 42 |

→ **10–14 entrenamientos**; a **8 paralelos** ≈ **2–3 h pared** vs ~20 h secuencial.

**Aceptación Fase 1:**

- Al menos un modelo v2 **igual o mejor** que incumbent en **recall@FP fijo** en val agrupado (presupuesto acordado, p.ej. FP/sesión equivalente a ≤10/día en replay ilustrativo).
- Sin regresión >2 pp recall en eventos **cortos** (<4 s visibles) vs incumbent en subset etiquetado.

---

### Fase 2 — Arquitectura y ensemble (P1/P2)

**Objetivo:** elegir **un** stack ligero estable (regla owner: preferir ligero si similar).

**Candidatos (solo tras Fase 1):**

1. **Masked TCN** (principal) — ~0,14M params  
2. **GRU pooled** — ~0,05–0,08M  
3. **ST-GCN** anatomical — comparador  
4. **Transformer pequeño** d64 — solo si TCN/GRU empatan  

**Ensemble v2:** **media de márgenes** de 3 semillas (no AND #41+#48 salvo comparador legacy).

**Jobs paralelos:** 4 arquitecturas × 3 seeds = **12 jobs** (+ 4 eval jobs) → con 8-wide ≈ **~2 h train + ~30 min eval**.

**Aceptación:** paired bootstrap en val agrupado; promover ligero si recall@FP a **≤2 pp** del mejor y menor latencia/memoria (benchmark `benchmark_serving.py` en campaign2).

---

### Fase 3 — Datos, hard negatives y bucle producción (P1 continuo)

**Objetivo:** mover la frontera con **información**, no con más grids.

**Implementación:**

- Manifest **`hard_negatives_v2.jsonl`** (FP revisados, near-miss, cat 14/3/4/2 enriquecidos).
- **`label_record_v2`:** `theft_status`, `concealment_motion` (ver respuesta experta §4.1).
- Re-train periódico: cada 2–4 semanas si ≥200 nuevos hard negs **distintos** (job único multi-seed en paralelo).

**Jobs paralelos:**

- Minado CPU (listados) || 3 seeds train en GPU.
- Eval paralelo por **día de tienda** (shards) para FP/día.

**Aceptación:** ≥20 % reducción relativa FP en días hold-out **o** ≥3 pp recall al mismo FP (criterio experto); si no, priorizar recolección antes de más arquitectura.

---

### Fase 4 — Post-procesado y handoff C2 (P2)

**Objetivo:** filtros **medidos** (no prometidos), sin sustituir un buen Stage 1.

Implementar en `campaign2/postprocess/` (eval offline, un filtro por job paralelo):

1. Dedup exacto ventana  
2. Cap 1 notificación / episodio + cooldown  
3. Abstención input inválido  
4. Persistencia 2-of-3 (solo si gana a max en igual FP)  
5. Handoff: JSON con márgenes, masks, `preprocess_hash` para SAM3  

**Paralelismo:** cada filtro = pipeline sobre **logits precomputados** (paralelo por shard de sesiones en CPU/GPU).

---

## 6. Configuración `campaign2_config.json` (esqueleto)

```json
{
  "version": 2,
  "description": "Kanvis campaign2 — windows, grouped splits, parallel GPU",
  "data_roots": [
    "/media/8TB/DatosEntrenamiento/data_clips/data_yolo26s_061026/data_result"
  ],
  "robbery_class": 6,
  "single_user_only": false,
  "target_track_policy": "user_cat_or_manifest",
  "temporal": {
    "window_seconds": 8.0,
    "target_hz": 12,
    "stride_seconds": 1.0,
    "seq_len": 96
  },
  "split": {
    "strategy": "grouped_chronological",
    "ratios": [0.75, 0.125, 0.125],
    "seed": 42,
    "group_keys": ["video_source", "session_component", "actor_id_when_known"],
    "embargo_seconds": 8.0
  },
  "parallelism": {
    "max_parallel_jobs": 8,
    "vram_reserve_mb": 2048,
    "auto_tune_from_probe": true,
    "probe_batch_sizes": [32, 64],
    "oom_retry_halve_batch": true,
    "max_dataloader_workers_per_job": 4,
    "max_cpu_workers_total": 32
  },
  "phases": {
    "p0_baseline": { "enabled": true, "incumbent_models": ["modelo_41", "modelo_48"] },
    "p1_train_v2": { "enabled": true, "job_matrix": "jobs/p1_matrix.json" },
    "p2_arch_compare": { "enabled": false },
    "p3_hard_neg": { "enabled": false },
    "p4_postprocess": { "enabled": false }
  },
  "selection": {
    "primary_metric": "event_recall_at_session_fp_budget",
    "fp_budget_per_1000_sessions": 10.0,
    "exploratory_fp_budget_per_1000_sessions": 30.0,
    "threshold_selection": "order_statistic_negative_sessions",
    "freeze_threshold_before_test": true
  },
  "seeds": [17, 42, 103]
}
```

---

## 7. Orquestación — flujo típico

```bash
cd IAModel/experiments/training/campaign2
export CAMPAIGN2_MAX_PARALLEL=8   # override opcional

./run_campaign2.sh new-run
./run_campaign2.sh probe          # mide VRAM/job, escribe parallel_jobs_recommended
./run_campaign2.sh preflight      # inventario + split agrupado + job list
./run_campaign2.sh train          # train_campaign2_parallel.py
./run_campaign2.sh eval           # evaluate_campaign2_parallel.py
./run_campaign2.sh summarize      # leaderboard + recomendación deploy
```

**Salida:** `artifacts/runs/<RUN_ID>/reports/campaign2_summary.json` + modelo promovido en `models/promoted/` con `preprocess_hash` y `threshold.json`.

---

## 8. Matriz de experimentos (evitar explosión combinatoria)

**No replicar** 8 celdas × 20 arquitecturas × multiclass/binario/full/filtered sin necesidad.

| Etapa | # jobs train (aprox.) | Paralelo 8 | Tiempo pared orientativo |
|-------|------------------------|------------|---------------------------|
| P0 eval only | 6–12 | 8 | <1 h |
| P1 core | 10–14 | 8 | 2–4 h |
| P2 arch | 12 | 8 | 2–3 h |
| P3 refresh | 3–6 | 3 | 1–2 h |

Comparado con **~2 días** secuenciales en v1 para un subset similar, **Campaign2 apunta a <1 día** para ciclo completo P0+P1+P2 con 8-wide, y **<6 h** cuando solo se re-entrenan 3 semillas TCN v2.

---

## 9. Criterios de éxito globales (definición de “entrenamos mejor”)

1. **Medición:** val/test **agrupado** + replay ventanas; FP reportado en **sesiones/día**, no solo % clips.  
2. **Mejora mínima vs incumbent (Fase 1):** recall evento **≥ +3 pp** al mismo FP/sesión **o** FP/sesión **≥ −25 % relativo** al mismo recall.  
3. **Estabilidad:** 3 semillas; desviación recall **≤ 3 pp** entre semillas.  
4. **Despliegue:** modelo **≤ 0,5M params** preferido si empata; latencia batch1 documentada en 5090.  
5. **Paralelismo:** utilización GPU media durante train **≥ 70 %** (varios procesos) medida con `nvidia-smi dmon` en logs.

---

## 10. Riesgos y mitigaciones

| Riesgo | Mitigación |
|--------|------------|
| OOM con muchos procesos | probe + auto_tune + retry batch |
| Contention CPU DataLoader | cap workers totales |
| Carreras al escribir modelos | un dir por job_id |
| Split leakage residual | auditoría de componentes conexas post-preflight |
| MIL mal configurado | ablation `ablation_no_mil` en paralelo con MISMA semilla |
| Expectativas 1 FP/día | documentar presupuesto exploratorio 10/día hasta ≥15k sesiones negativas |

---

## 11. Orden de trabajo para desarrolladores (checklist)

**Sprint 1 — Infra paralela**

- [ ] `campaign2/` + `campaign2_config.json`  
- [ ] `gpu_scheduler.py` + `gpu_probe_campaign2.py`  
- [ ] `train_campaign2_parallel.py` (spawn, cola, logs)  
- [ ] `run_campaign2.sh` (fases)

**Sprint 2 — P0**

- [ ] Loader ventanas sobre npy legacy (sin re-extracción)  
- [ ] `evaluate_campaign2` + jobs incumbent paralelos  
- [ ] Informe baseline

**Sprint 3 — P1 train_v2**

- [ ] `dataset_windows.py`, `normalize.py`, masked TCN  
- [ ] Split agrupado en preflight  
- [ ] Matriz P1 + 3 seeds en paralelo  
- [ ] Selección por recall@FP

**Sprint 4 — P2/P3/P4**

- [ ] Comparadores arquitectura  
- [ ] Hard neg manifest + retrain  
- [ ] Postprocess eval paralelo + handoff C2

---

## 12. Referencias cruzadas (decisiones de entrenamiento)

Resumen de **qué implementa este plan** respecto al experto:

| Tema | Decisión Campaign2 |
|------|---------------------|
| Ventanas vs clip entero | Ventanas 8s/12Hz en train y eval |
| #41+#48 | Solo baseline P0; producción v2 = mean margin 3 seeds TCN |
| H1 | Corregido en v2 |
| single_user_only | false + track objetivo |
| Umbral | Congelado; order-statistic en sesiones negativas (T) |
| Etiqueta móvil | Campos theft/concealment; revisión FP |
| Paralelización | Scheduler multi-job GPU (hasta ~8–10 TCN) |

---

## 13. Historial

| Versión | Fecha | Notas |
|---------|-------|-------|
| 1.0 | 2026-10-16 | Plan maestro inicial: fases 0–4 + paralelización total 5090 |

---

*Fin del documento. Implementación código: `IAModel/experiments/training/campaign2/` (pendiente según checklist §11).*
