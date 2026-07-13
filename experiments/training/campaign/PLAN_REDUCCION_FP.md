# Plan de reducción de falsos positivos — GuadIA robo binario

**Run de referencia:** `mass_20260707_121819`  
**Mejor config actual:** celda `bin_filtered_hardened`, ensemble `modelo_36|modelo_40`, regla MEAN, umbral **0,68**  
**Val real:** 1 200 clips · 192 robos · 1 008 no-robos  
**Métricas actuales:** F1 73,7 % · recall 62,0 % (119/192) · **12 FP** (1,19 % de no-robos) · **73 FN**  
**Objetivo config:** FP ≤ 0,01 % (~0–1 FP en val) manteniendo recall ≥ 60 %  

---

## Diagnóstico

1. **Mass augment (~100k filas sintéticas)** no mejoró de forma operativa FP/FN respecto a entrenar solo con reales → más volumen del mismo tipo tiene rendimientos decrecientes.
2. Los FP se concentran en categorías **2, 3 y 14** (~66 % del patrón de confusión).
3. Pasar de **12 FP → ~0–1 FP** es una reducción ~100×; no es realista solo con más datos genéricos o un umbral más alto sin perder recall.
4. El VLM “objeto en mano” falló (~50 %) porque la pregunta visual es inestable; **cinemática de pose** es más alineada con lo que ya entrenáis.

---

## Respuestas a decisiones de diseño

### ¿Verificador por tipo 2 / tipo 3 vs multiclase?

**No** se trata de entrenar un modelo solo para cat 3 y otro para cat 2 que “reclasifiquen” tras la alarma.

| Enfoque | Qué hace | Problema / ventaja |
|---------|----------|-------------------|
| **Multiclase (`mc_*`)** | Elige entre todas las clases (0…N) | Capacidad repartida; en val da FP ~5–6 % — peor para despliegue binario |
| **Un modelo por cat (2, 3, 14…)** | N verificadores | Coste de mantenimiento alto; cats 2 y 3 comparten gestos similares |
| **Verificador binario especializado (recomendado)** | **Robo vs “confusable”** (pool {2, 3, 14}) | Toda la capacidad en el límite que os duele; una sola 2ª etapa |

**Pipeline recomendado:**

```
Etapa 1 (actual):  P_robo ≥ 0,68  →  candidato a alarma
Etapa 2 (nueva):   P(confusable)   →  si alta, DESCARTAR alarma
                   entrenada solo robo vs clips de cats 2/3/14 (+ FP exportados)
Etapa 3 (opcional): heurísticas de pose (ver sección 5)
```

La etapa 2 **no sustituye** al multiclase; es un **filtro barato** entrenado en el subproblema difícil.

---

### Regla temporal con ventanas de 8 s

Tienes razón: **comprar y robar pueden durar lo mismo (~3 s)** dentro de una ventana de 8 s. La regla temporal **no sirve para distinguir compra vs robo**.

Úsala solo para:

- **Estabilidad de la score del modelo:** exigir P(robo) ≥ umbral en ≥ K frames consecutivos (suavizado), no un pico de 1–2 frames.
- **Secuencia cinemática dentro del clip:** exigir patrón **reach → conceal** (o similar) en una ventana temporal coherente — esto sí separa muchos FP de gestos sueltos.

**No usar:** “varias ventanas de 8 s seguidas” si cada clip es ya una sola ventana aislada.

---

## Fase 0 — Sin reentrenar (1–2 días)

### 0.1 Revisar los 12 FP en vídeo

En angel:

```bash
cat ~/IAModel/experiments/training/campaign/artifacts/runs/mass_20260707_121819/reports/bin_filtered_hardened/val_fp_ensemble_mean_modelo_36+modelo_40_t0.680+0.680.txt
```

Anotar por clip: `folder_category`, gesto visible, ¿reach+conceal?, ¿error de etiqueta?

### 0.2 Probar ensembles más conservadores (sin reentrenar)

Los checkpoints ya existen. Opciones:

**A) Re-evaluar solo inferencia** (regenera grid de ensembles):

```bash
cd ~/IAModel/experiments/training/campaign
./run_campaign.sh eval-fg --run-id mass_20260707_121819 --cells bin_filtered_hardened
```

Salida clave: `artifacts/runs/mass_20260707_121819/reports/bin_filtered_hardened/val_ensemble_grid.csv`

Filtrar candidatos conservadores:

```bash
# Ejemplo: FP bajo y recall ≥ 58 %
python3 - <<'PY'
import csv
from pathlib import Path
p = Path("artifacts/runs/mass_20260707_121819/reports/bin_filtered_hardened/val_ensemble_grid.csv")
rows = list(csv.DictReader(p.open()))
ok = [r for r in rows
      if float(r["false_positive_rate_pct"]) <= 0.5
      and float(r["recall_robbery_pct"]) >= 58]
ok.sort(key=lambda r: (-float(r["f1_pct"]), float(r["false_positive_rate_pct"])))
for r in ok[:15]:
    print(r["decision_mode"], r["models"], r["thresholds"],
          "F1", r["f1_pct"], "recall", r["recall_robbery_pct"], "FP%", r["false_positive_rate_pct"])
PY
```

**B) Probar regla AND / cascade explícita** (export FP/FN sin retrain):

```bash
# AND más estricto (ejemplo)
python export_ensemble_fp.py --run-id mass_20260707_121819 \
  --cell bin_filtered_hardened --split val \
  --models modelo_36 modelo_40 --rule and --threshold 0.72

# Cascade (modelo permisivo + estricto)
python export_ensemble_fp.py --run-id mass_20260707_121819 \
  --cell bin_filtered_hardened --split val \
  --models modelo_40 modelo_36 --rule cascade
```

**C) Barrido manual de umbrales MEAN** (mismo par 36|40):

| Umbral MEAN | Efecto esperado |
|-------------|-----------------|
| 0,68 (actual) | Baseline: 12 FP, recall 62 % |
| 0,72 – 0,76 | Menos FP, recall ↓ |
| 0,80+ | FP muy bajos, recall puede caer por debajo del 60 % |

Registrar curva FP–recall y elegir punto operativo antes de reentrenar.

---

## Fase 1 — Hard negatives curados (1 semana)

### 1.1 Datos a añadir (prioridad sobre “30k reales genéricos”)

| Prioridad | Tipo | Objetivo |
|-----------|------|----------|
| P0 | Los **12 FP** del ensemble + re-export periódico | Corregir errores concretos |
| P1 | Clips reales **cats 2, 3, 14** (más diversidad) | Endurecer frontera |
| P2 | Clips ambiguos revisados manualmente | Limpiar etiqueta |
| P3 | Más robos reales | Sobre todo **FN**, no FP |

### 1.2 Reentrenamiento con perfil HN

Usar `campaign_config_improve.json` / perfil `fp_hardened_hn`:

- Peso alto en uids de FP exportados (`hard_negative_csv`).
- **No** repetir mass augment ancho; synth solo como variante de FP y cats 2/3/14.

### 1.3 Criterio de éxito fase 1

- FP en val ≤ **5** (≤ 0,5 %) con recall ≥ **58 %**, **o**
- FP ≤ **3** aceptando recall ≥ **55 %** (decisión de producto).

---

## Fase 2 — Verificador robo vs confusables (1–2 semanas)

### 2.1 Dataset

- **Positivos:** todos los robos (clase 6) del train.
- **Negativos:** solo cats **2, 3, 14** + uids de FP históricos.
- Misma pipeline de poses (`filtered`, `fp_hardened` augment ligero).

### 2.2 Modelo

- Binario pequeño (TCN/GRU) o mismo zoo con `--cells` dedicada, p. ej. `bin_verifier_234`.
- Salida: `P_confusable`.

### 2.3 Regla de despliegue

```
ALARMA final = (Etapa1 dispara) AND (P_confusable < τ_verify)
```

Calibrar `τ_verify` en val (no en test).

### 2.4 Criterio de éxito fase 2

- FP ≤ **2–3** manteniendo recall global ≥ **58 %**.

---

## Fase 3 — Filtro heurístico de pose (post-proceso general)

Base de código existente: `experiments/training/clean_theft.py` (`reach`, `conceal`, `low_pick`).

Keypoints (8): hombros 0/1, muñecas 4/5, caderas 6/7. Coordenadas normalizadas [0,1].

### 3.1 Catálogo de señales (por ventana de 8 s)

| ID | Señal | Descripción | Típico en robo | Típico en FP |
|----|-------|-------------|----------------|--------------|
| H1 | **reach_peak** | Extensión brazo (muñeca–hombro) / altura torso | Alto en coger | Variable |
| H2 | **conceal_peak** | Muñeca cerca cadera/torso | Alto al ocultar | Bajo en gestos abiertos |
| H3 | **low_pick_peak** | Alcance hacia abajo (carrito) | Algunos robos | Compras normales |
| H4 | **reach_then_conceal** | Pico reach seguido de pico conceal (≤ N frames, p. ej. 45) | Muy frecuente en robo | Raro en FP “gesto suelto” |
| H5 | **front_pocket** | Muñeca cerca cadera **delantera** (hip ipsilateral, y ≥ cadera) | Ocultación delantera | Depende del FP |
| H6 | **back_pocket** | Muñeca cerca cadera **trasera** + muñeca detrás del plano torso | Ocultación trasera | Depende del FP |
| H7 | **torso_proximity_duration** | Frames con muñeca–torso < umbral (normalizado) | Sostenido en ocultar | Breve o nulo |
| H8 | **retraction** | Tras reach, distancia muñeca–hombro cae rápido | Tras meter en bolsillo | Menos marcado |
| H9 | **asymmetry** | Solo un brazo activo (reach/conceal lateral) | Común | Común también |
| H10 | **pose_quality** | % frames con 8 keypoints visibles | Gate: baja calidad → no alarmar | Evita FP por ruido |
| H11 | **model_score_stability** | P(robo) suavizada ≥ umbral durante ≥ K frames | Estable si robo claro | Picos aislados |
| H12 | **combined_theft_signal** | max(reach, conceal, low_pick) de `clean_theft` | Alto | Bajo–medio |

### 3.2 Regla de decisión propuesta (scoring, no una sola heurística)

Calcular score **`S_robo_kin`** ∈ [0, 1]:

```
S_robo_kin = w1·norm(H1) + w2·norm(H2) + w3·norm(H4) + w4·norm(H7) + w5·norm(H12)
```

**Descartar alarma** (tratar como no-robo) si:

```
Etapa1 = alarma  AND  S_robo_kin < T_kin
```

**Confirmar alarma** (refuerzo) si:

```
(H4 alto) OR (H2 alto AND (H5 OR H6)) OR (H1 alto AND H7 alto)
```

Calibrar `T_kin` y pesos en **val**:

- **Robos reales (192):** objetivo ≥ **85–90 %** con `S_robo_kin ≥ T_kin` o regla de confirmación.
- **12 FP:** objetivo ≥ **80–90 %** clasificados como `S_robo_kin < T_kin`.

### 3.3 Implementación sugerida

1. Script `pose_robbery_heuristics.py`: entrada `poses.npy` → JSON con H1…H12 + `S_robo_kin`.
2. Batch sobre val: unir con CSV de `export_ensemble_fp.py`.
3. Barrido de `T_kin` para curva FP–recall **después** del ensemble.
4. Integrar en inferencia solo si mejora val sin retrain del TCN principal.

### 3.4 Qué **no** esperar de las heurísticas

- No distinguirán bien **compra vs robo** si ambos tienen reach similar **sin** fase de conceal.
- Ahí la señal clave es **H4 + H2/H5/H6**, no la duración total.

---

## Fase 4 — Política operativa (si FP ≤ 0,01 % sigue inalcanzable)

Si tras fases 0–3 seguís en FP ~0,3–0,5 %:

| Política | Descripción |
|----------|-------------|
| **Zona gris** | 0,55 ≤ P < 0,75 → no contar como alarma dura / revisión |
| **Doble confirmación** | Alarma solo si Etapa1 + verificador + heurística |
| **Métrica de producto** | “Alarmas accionables/día” en lugar de FP rate teórico 0,01 % |

---

## Orden de ejecución recomendado

```
Semana 1
  ├─ Fase 0.1  Ver vídeos 12 FP
  ├─ Fase 0.2  Barrido ensemble AND / umbrales / cascade (sin retrain)
  └─ Documentar curva FP–recall

Semana 2
  ├─ Fase 1    Hard negatives (12 FP + cats 2/3/14)
  └─ Re-eval val

Semana 3–4
  ├─ Fase 2    Verificador robo vs {2,3,14}
  └─ Fase 3    Prototipo heurísticas (clean_theft + H4/H5/H6)

Semana 5
  └─ Integración + test holdout (1215 clips) con config congelada
```

---

## Métricas de aceptación por fase

| Fase | FP (de 1008 no-robos) | FP rate | Recall (de 192) | Acción si no cumple |
|------|----------------------:|--------:|------------------:|---------------------|
| Baseline | 12 | 1,19 % | 62,0 % | — |
| 0 (ensemble conservador) | ≤ 8 | ≤ 0,8 % | ≥ 58 % | Subir umbral / AND |
| 1 (HN) | ≤ 5 | ≤ 0,5 % | ≥ 58 % | Más FP curados |
| 2 (+ verificador) | ≤ 3 | ≤ 0,3 % | ≥ 56 % | Ajustar τ_verify |
| 3 (+ heurísticas) | ≤ 1–2 | ≤ 0,2 % | ≥ 55 % | Revisar política producto |
| Objetivo config | ~0–1 | 0,01 % | ≥ 60 % | Puede requerir fase 4 |

---

## Expectativa sobre “30k clips reales más”

| Escenario | Impacto esperado en FP |
|-----------|------------------------|
| 30k reales **aleatorios** | Baja–moderada (similar a mass aug) |
| 30k reales **cats 2/3/14** + FP minados | Alta |
| Solo reentrenar con más robos | Mejora **FN**, poco efecto en **FP** |

**Conclusión:** más reales **sí ayudan**, pero solo si son **hard negatives dirigidos**, no volumen genérico. Combinar con verificador + heurísticas tiene más probabilidad de acercarse a FP ~0 que solo escalar dataset.

---

## Comandos rápidos (referencia)

```bash
RUN=mass_20260707_121819
CELL=bin_filtered_hardened
cd ~/IAModel/experiments/training/campaign

# Listar FP actuales
cat artifacts/runs/$RUN/reports/$CELL/val_fp_ensemble_mean_modelo_36+modelo_40_t0.680+0.680.txt

# Re-export con otro ensemble
python export_ensemble_fp.py --run-id $RUN --cell $CELL --split val \
  --models modelo_36 modelo_40 --rule and --threshold 0.74 --outcomes errors

# Re-eval (ensemble grid, sin train)
./run_campaign.sh eval-fg --run-id $RUN --cells $CELL
```

---

## Próximo paso inmediato

1. Ver los **12 FP** y clasificarlos (¿tienen fase conceal? ¿cat?).
2. Ejecutar **Fase 0.2** (AND @ 0,72–0,76, cascade 40→36) y anotar FP/recall.
3. Paralelamente: prototipo **`pose_robbery_heuristics.py`** sobre val completo y medir separación robos vs 12 FP.

---

## Implementación en repo (2026-07-13)

### Artefactos nuevos

| Fichero | Rol |
|---------|-----|
| `pose_robbery_heuristics.py` | Heurísticas H1–H12; patrones **reach→conceal** y **conceal-only** (robo en pasillo) |
| `merge_verifier_probs.py` | Añade `p_verifier` al CSV del ensemble |
| `campaign_config_fp_pipeline.json` | Train sin mass aug: etapa 1 + verificador |
| `class_maps/map_verifier_confusable.json` | Solo cats 6 vs 2/3/14 |
| `run_fp_pipeline.sh` | Orquestador del pipeline completo |

### Patrón conceal-only (sin H1 previo)

El score `conceal_only` detecta ocultación fuerte **sin reach dominante previo** (objeto ya en mano, ocultar en pasillo). El score global `s_kin` usa:

```
s_kin = max(reach→conceal, conceal_only, 0.65·reach + 0.35·conceal, low_pick)
```

### Ejecución completa en angel

```bash
cd ~/IAModel/experiments/training/campaign

# 1. Hard negatives del run anterior (opcional pero recomendado)
export RUN_ID=fp_pipeline_v1
export HN_CSV=artifacts/runs/mass_20260707_121819/reports/bin_filtered_hardened/val_ensemble_fp_mean_modelo_36+modelo_40_t0.68.csv

# 2. Preflight + train (2 celdas, augment ligero uniform_ops=2, sin mass augment)
./run_fp_pipeline.sh preflight
./run_fp_pipeline.sh train          # o: ./run_fp_pipeline.sh all-bg

# 3. Eval (genera val_ensemble_grid.csv, best_ensemble.json, FP lists)
./run_fp_pipeline.sh eval

# 4. Export ensemble etapa 1 (ajusta modelos tras ver best_ensemble.json)
./run_fp_pipeline.sh export-stage1 \
  --models modelo_36 modelo_40 --rule mean --threshold 0.68

# 5. Probar ensembles conservadores SIN reentrenar (paralelo al paso 4)
python export_ensemble_fp.py --run-id $RUN_ID --cell bin_filtered_hardened --split val \
  --models modelo_36 modelo_40 --rule and --threshold 0.74 --outcomes errors

# 6. Merge verificador etapa 2
ENS=artifacts/runs/$RUN_ID/reports/bin_filtered_hardened/val_ensemble_fp_mean_modelo_36+modelo_40_t0.68.csv
./run_fp_pipeline.sh merge-verifier "$ENS"

# 7. Barrido pipeline: etapa1 + p_verifier + heurísticas + regla temporal cinemática
./run_fp_pipeline.sh pipeline-sweep \
  artifacts/runs/$RUN_ID/reports/bin_filtered_hardened/val_ensemble_fp_mean_modelo_36+modelo_40_t0.68_with_verifier.csv \
  --rule and

# 8. Heurísticas standalone (todo val)
./run_fp_pipeline.sh heuristics-batch
```

### Salidas a revisar

| Fichero | Contenido |
|---------|-----------|
| `reports/bin_filtered_hardened/val_ensemble_grid.csv` | Barrido AND/MEAN/cascade sin retrain |
| `reports/bin_filtered_hardened/fp_pipeline/pipeline_sweep.csv` | FP/FN/F1 por combinación t_stage1 × t_kin |
| `reports/bin_filtered_hardened/fp_pipeline/pipeline_best_summary.json` | Mejor combo pipeline completo |
| `reports/bin_verifier_234/val_best_ensemble.json` | Mejor modelo verificador |
| `reports/bin_filtered_hardened/val_heuristics_features.csv` | s_kin, pattern, conceal_only por clip |

### Criterios de éxito del nuevo run

| Métrica | Baseline mass aug | Objetivo fp_pipeline_v1 |
|---------|-------------------|-------------------------|
| FP (val) | 12 | ≤ 5 (fase 1) → ≤ 3 (con verificador+heur) |
| FN (val) | 73 | ≤ 80 (aceptar ligero ↑ si FP baja mucho) |
| FP rate | 1,19 % | ≤ 0,5 % |

### Test holdout

Repetir pasos 4–7 con `--split test` **solo cuando** la config esté congelada en val.
