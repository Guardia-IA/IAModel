# Interpretación — evaluación sliding window (anti-FP)

**RUN_ID:** `campaign_20260714_164642`  
**Fecha análisis:** 2026-07-15  
**Split:** val (1241 clips)  
**Ventana:** 3.0 s, stride 1.0 s, fps 12  

> Copia en repo. Artefactos de ejecución en:  
> `/home/debian/Documentos/Kanvis/artifacts/runs/campaign_20260714_164642/`

---

## 1. Objetivo de la batería

Reducir falsos positivos (FP) usando inferencia por ventanas deslizantes + filtros temporales y cinemáticos, manteniendo aceptable el recall de robos. Meta aspiracional del usuario: **FP ≈ 1** aunque el F1 baje de ~88% a ~65%.

Scripts involucrados:

- `experiments/training/campaign/sliding_window_eval.py`
- `experiments/training/campaign/run_sliding_window_eval.sh`
- `experiments/training/campaign/run_campaign.sh` (comandos `sliding-window-*`)

Artefactos generados:

```
artifacts/runs/campaign_20260714_164642/
├── logs/sliding_window_battery.log
├── reports/bin_full/
│   ├── val_sliding_window_modelo_12_*
│   ├── val_sliding_window_ensemble_mean_modelo_11+modelo_12_t0.50_*
│   ├── val_sliding_window_ensemble_and_modelo_11+modelo_40_t0.74+0.74_*
│   └── val_sliding_window_combined_summary.json
└── reports/mc_full/
    └── val_sliding_window_modelo_12_*
```

---

## 2. Veredicto principal

**La hipótesis temporal funciona parcialmente, pero el objetivo de ~0 FP no se alcanza.**

- En **ningún predictor** ni configuración del barrido aparece `FP ≤ 1`.
- Mínimo en subconjunto fp-only (81 clips = FP+FN del modelo): **FP = 15** (`modelo_12` binario).
- Mínimo en val completo (1241 clips): **FP = 11** (`modelo_12` multiclase, barrido agresivo).

**Referencia campaña original** (`modelo_12` argmax clip completo, `bin_full`): **FP = 47, FN = 33, F1 ≈ 86%, recall ≈ 88%**.

> **Nota:** El “~88%” que a menudo se recuerda puede ser el **recall** del `modelo_12` (~88%), el **F1 del ensemble 11|12 mean** (~87.8%), o la **accuracy** del split (~93.5%) — no el F1 del modelo solo.

---

## 3. Campaña original vs sliding window — baseline (sin filtro temporal)

Esta tabla aclara que el **`modelo_12` binario sí está evaluado**: la columna “Sliding window baseline” es la inferencia clip-completo **antes** de aplicar filtros por ventanas. Debe coincidir con la eval de campaña si el modo de decisión es el mismo.

Fuente campaña: `eval_pack_campaign_20260714_164642/bin_full/val_leaderboard.csv` y `val_eval_summary.json`.  
Fuente sliding window: `logs/sliding_window_battery.log` (paso 1/3, 1241 clips).

### 3.1 Modelos y modos de decisión (eval campaña, clip completo)

| Predictor | Modo decisión | F1 | Recall | FP | FN | TP | Notas |
|---|---|---:|---:|---:|---:|---:|---|
| **modelo_12** | argmax (softmax) | **85.9%** | **87.9%** | 47 | 34 | 247 | **Mejor modelo solo** (`bin_full`) |
| **modelo_12** | umbral softmax @ 0.8 | 83.3% | 79.7% | 33 | 57 | 224 | Menos FP, más FN |
| **modelo_12** | logit_margin @ 0.0 | 85.9% | 87.9% | 47 | 34 | 247 | Igual que argmax en este run |
| **11\|12 mean** | ensemble @ 0.50 | **87.8%** | **88.3%** | 36 | 33 | 248 | **Mejor F1**; suele confundirse con “el 12” |
| **11\|12 AND** | ensemble @ 0.50 | 84.3% | 79.0% | 24 | 59 | 222 | Más conservador que mean |
| **11\|40 AND** | ensemble @ 0.74 | 74.7% | 60.9% | **6** | 110 | 171 | **Mejor low-FP** en grid campaña |

### 3.2 ¿Coincide el baseline del sliding window con la campaña?

| Predictor | F1 campaña | F1 SW baseline | FP campaña | FP SW baseline | ¿Coincide? |
|---|---:|---:|---:|---:|---|
| **modelo_12 binario argmax** | 85.9% | 85.9% | 47 | 47 | **Sí** |
| **modelo_12 multiclase argmax** | 87.3% | 87.3% | 38 | 38 | **Sí** (celda `mc_full`) |
| **11\|12 mean @ 0.50** | 87.8% | — | 36 | — | No evaluado en SW full* |
| **11\|40 AND @ 0.74** | 74.7% (FP=6) | 84.6% (FP=42) | 6 | 42 | **No** — ver nota abajo |

\*En sliding window, el ensemble 11|12 se evaluó sobre el subconjunto fp-only (81 clips) y en una pasada full con baseline distinto (FP=52); no reprodujo exactamente el eval de campaña clip-completo.

**Discrepancia ensemble 11|40:** En campaña, FP=6 con umbral AND @ 0.74 sobre clip completo. En sliding window, el baseline del ensemble usa otra lógica de alarma (argmax clase 6 vs umbral en ventanas), por eso sale FP=42. **Comparar siempre con la misma metodología.**

### 3.3 Lectura rápida — ¿qué número es cuál?

| Si recuerdas… | En realidad es… |
|---|---|
| “F1 ~88%” | Probablemente **ensemble 11\|12 mean** (87.8%), no `modelo_12` solo |
| “Recall ~88%” | **modelo_12 argmax** (87.9%) o ensemble 11\|12 mean (88.3%) |
| “Recall ~80%” | **modelo_12 umbral 0.8** (79.7%) o ensemble 11\|12 AND (~79%) |
| “FP = 47” | **modelo_12 argmax** — baseline de la fila principal en sliding window |
| “FP = 6” | **11\|40 AND @ 0.74** en eval campaña (no en SW baseline) |

---

## 4. Tabla comparativa — efecto del filtro temporal (val completo, 1241 clips)

La columna **“Baseline FP”** de esta tabla **es el modelo sin filtro** (equivalente a campaña para `modelo_12` argmax). Las columnas siguientes muestran cuánto bajan los FP **después** de ventanas + heurísticas.

| Configuración | Baseline FP | Tras filtro (default) | Mejor barrido FP | F1 tras filtro | Recall tras filtro |
|---|---:|---:|---:|---:|---:|
| **modelo_12 binario** (`bin_full`) | 47 | **28** (−19) | **15** | 72.3% → 52.4%* | 62.3% → 37.4%* |
| **modelo_12 multiclase** (`mc_full`) | 38 | **24** (−14) | **11** | 72.4% → 50.6%* | 61.6% → 35.2%* |
| **ensemble 11\|12 mean @ 0.50** | 52 | **33** (−19) | **17** | 73.1% → 63.0%* | 64.4% → 48.8%* |
| **ensemble 11\|40 AND @ 0.74** | 42 | **22** (−20) | **16** | 63.5% | 50.2% |

\*Tras barrido agresivo de políticas (`min_cons=3`, `min_kin≥0.58`, `p_win=0.65`, etc.).

### Política default aplicada

```yaml
window_sec: 3.0
stride_sec: 1.0
p_window_threshold: 0.5
full_clip_threshold: 0.5
alarm_mode: filter_baseline
min_consecutive_windows: 2
post_purchase_veto_windows: 3
purchase_classes: [3, 4, 5]
require_robbery_like: true
min_s_kin: 0.42
min_pose_quality: 0.55
veto_isolated_spike: true
```

---

## 5. Resultados detallados por paso de la batería

### 5.1 Paso 1 — `bin_full` full + sweep (`modelo_12` binario, 1241 clips)

| Métrica | Baseline | Filtrado (default) | Mejor barrido |
|---|---:|---:|---:|
| TP | 247 | 175 | — |
| FP | 47 | 28 | **15** |
| FN | 34 | 106 | 176 |
| F1 | 85.9% | 72.3% | 52.4% |
| Recall | 87.9% | 62.3% | 37.4% |
| FP eliminados | — | 19 | 32 |
| TP perdidos | — | 72 | 142 |

Política del mejor barrido: `min_cons=3 min_kin=0.58 p_win=0.65 full_thr=0.4 mode=filter_baseline`

### 5.2 Paso 2 — fp-only + sweep (`modelo_12` binario, 81 clips de error)

Solo clips FP+FN del baseline. Métricas globales no representan val completo.

| Métrica | Baseline | Filtrado (default) | Mejor barrido |
|---|---:|---:|---:|
| FP | 47 | 28 | **15** |
| FN | 34 | 34 | 26 |
| FP eliminados | — | 19 | 32 |
| TP perdidos | — | 0 | 0 |

En fp-only, el barrido agresivo **no pierde TP** (`tp_lost=0`) porque solo filtra alarmas que ya eran FP.

### 5.3 Paso 3 — `mc_full` multiclass + sweep (`modelo_12`, 1241 clips)

| Métrica | Baseline | Filtrado (default) | Mejor barrido |
|---|---:|---:|---:|
| TP | 247 | 173 | — |
| FP | 38 | 24 | **11** |
| FN | 34 | 108 | 182 |
| F1 | 87.3% | 72.4% | 50.6% |
| Recall | 87.9% | 61.6% | 35.2% |
| FP eliminados | — | 14 | 27 |
| TP perdidos | — | 74 | 148 |

El multiclase parte con menos FP baseline (38 vs 47) y el veto post-compra sí actúa.

### 5.4 Ensemble F1 — `11|12 mean @ 0.50` (fp-only, 81 clips)

| Métrica | Baseline | Filtrado | Mejor barrido |
|---|---:|---:|---:|
| TP | 15 | 9 | — |
| FP | 34 | 20 | **10** |
| FN | 19 | 25 | 29 |
| F1 | 36.1% | 28.6% | 20.4% |

En val completo (segunda pasada del log): baseline FP=52, filtrado FP=33, mejor barrido FP=17.

### 5.5 Ensemble conservador — `11|40 AND @ 0.74` (1241 clips)

| Métrica | Baseline | Filtrado (default) | Mejor barrido |
|---|---:|---:|---:|
| TP | 237 | 141 | — |
| FP | 42 | 22 | **16** |
| FN | 44 | 140 | 181 |
| F1 | 84.6% | 63.5% | 50.4% |
| Recall | 84.3% | 50.2% | 35.6% |
| FP eliminados | — | 20 | 26 |
| TP perdidos | — | 96 | 137 |

**Nota:** En eval clip-completo de campaña, este ensemble tenía **FP = 6** (`val_best_ensemble.json`). En sliding window el baseline muestra FP=42 porque la decisión baseline del ensemble en ventanas puede diferir del eval original (umbral vs argmax). Comparar siempre contra la misma metodología.

---

## 6. ¿Qué filtros funcionan?

### 6.1 Veto de pico aislado (`no_consecutive_windows`)

- **Principal contribución** a la reducción de FP.
- Elimina alarmas donde solo 1 ventana (o menos de 2 consecutivas) supera umbral.
- En fp-only binario: **19/19 eliminaciones** usan esta razón.
- Encaja con la hipótesis: gesto puntual confundido con robo dentro de un clip largo.

### 6.2 Veto post-compra (`post_purchase_veto`)

- Funciona **solo con modelo multiclase** en ventanas.
- En `mc_full`: **6 FP eliminados** con esta razón.
- En binario puro: **0 eliminaciones** (no hay predicción de clases 3/4/5 por ventana).
- Valida parcialmente la regla: *“si tras un 6 viene compra (3/4/5), no es robo”*.

### 6.3 Filtros cinemáticos (`min_s_kin`, `require_robbery_like`)

- En política default eliminan poco de forma aislada.
- En barrido agresivo (`min_kin ≥ 0.58`) contribuyen a bajar FP pero con fuerte coste en recall.
- Los FP que **sobreviven** pasan todos los filtros con razón `alarm`.

---

## 7. FP duros — perfil de los que no se eliminan

Tras filtro default en fp-only (`modelo_12`, 28 FP restantes). Todos con `filter_reason = alarm`.

| Clase carpeta | FP restantes | Interpretación |
|---:|---:|---|
| **2** (mirar estantería) | 10 | Gestos de alcance muy parecidos al robo |
| **14** (piloto Kanvis) | 8 | Dominio/cámara distinta, alta confianza del modelo |
| **3** (compra) | 7 | Modelo ve robo **persistente**, no pico + compra |
| **1** | 1 | Caso suelto |
| **7** | 1 | Caso suelto |
| **9** | 1 | Caso suelto |

### FP eliminados — perfil (19 clips, binario fp-only)

| Clase | Eliminados |
|---:|---:|
| 0 | 5 |
| 5 | 5 |
| 4 | 3 |
| 2 | 3 |
| 9 | 2 |
| 14 | 1 |

Mayoría: clips cortos de clases neutras/compra donde el 6 era un pico aislado.

---

## 8. ¿Se cumple el objetivo (F1 ~65% con FP ≈ 1)?

**No.**

| Objetivo usuario | Mejor resultado observado |
|---|---|
| FP ≈ 1 | **Ninguna config** en barrido (`meets_fp_target=False` en todos los CSV) |
| F1 ~65% con FP mínimo | Ensemble AND default: F1=63.5%, **FP=22** |
| FP ≤ 15 | F1≈52%, recall≈37% (modelo_12 binario, barrido full) |

El post-procesado temporal reduce FP **~40–50%**, pero no sustituye un modelo/ensemble más conservador o más datos de entrenamiento para casi cero FP.

---

## 9. Recomendaciones

### Opción A — Despliegue intermedio (recomendada si se quiere bajar FP ya)

- **Predictor:** `modelo_12` binario (`bin_full`)
- **Filtro:** política default (ventana 3s, 2 ventanas consecutivas, veto pico aislado)
- **Resultado:** 47 → **28 FP**, F1 **86% → 72%**, recall **88% → 62%**
- **Coste:** ~72 robos dejan de detectarse vs baseline

### Opción B — Máxima reducción con heurísticas actuales

- **Predictor:** `modelo_12` multiclase + `post_purchase_veto`
- **Resultado:** FP=11 en val completo (barrido), F1≈51%, recall≈35%
- Solo viable si se priorizan alarmas falsas sobre detectar robos

### Opción C — Camino hacia FP ≈ 0

1. Combinar **ensemble conservador** (p.ej. `49|57 mean @ 0.86` → 6 FP en eval clip completo) **+** filtro temporal encima.
2. Revisar manualmente los **15–28 FP duros** listados en:
   - `reports/bin_full/val_sliding_window_modelo_12_fp_remaining.txt`
3. Activar veto post-compra con **modelo multiclase en ventanas**, no binario.
4. Reglas específicas para clases problemáticas (14, 2, 3 persistente).

---

## 10. Notas técnicas

### 10.1 JSON sobrescrito en `bin_full`

`reports/bin_full/val_sliding_window_modelo_12_summary.json` refleja la **última pasada fp-only** (81 clips), no val completo. Para análisis global usar:

- Log: `logs/sliding_window_battery.log` (paso 1/3, 1241 clips)
- `reports/mc_full/val_sliding_window_modelo_12_summary.json` (1241 clips, multiclase)

### 10.2 Archivos clave para inspección

| Archivo | Contenido |
|---|---|
| `*_fp_remaining.txt` | UIDs de FP que sobreviven al filtro |
| `*_fp_removed.txt` | UIDs de FP eliminados |
| `*_tp_lost.txt` | Robos reales que dejaron de detectarse |
| `*_clips.csv` | Detalle por clip: `filter_reason`, `baseline_p_robo`, ventanas |
| `*_sweep.csv` | Barrido de políticas (top 500 configs) |

### 10.3 Hipótesis del usuario — validación

> *“El modelo evalúa todo el clip; un gesto de compra en una ventana puede disparar robo aunque el clip completo no lo sea. Si tras un 6 viene 3/4, no debería ser robo.”*

- **Confirmada parcialmente:** muchos FP eliminados son picos aislados en clips de compra/neutros.
- **Limitación:** los FP duros muestran señal de robo **persistente** en ventanas consecutivas; el veto post-compra no aplica si no hay clase compra posterior o el 6 domina todo el clip.

---

## 11. Próximo paso sugerido

Informe clip a clip de los **15–28 FP duros** con:

- Secuencia de clases por ventana
- `filter_reason` y `s_kin` por ventana
- Clasificación: “eliminable con regla nueva” vs “error del modelo (requiere reentrenar/ensemble)”

Comando útil para relanzar evaluación:

```bash
./run_campaign.sh sliding-window-battery-bg --run-id campaign_20260714_164642
```

---

## 12. Contexto de conversación previa

- Comparativa yolo26n vs yolo26m: no había eval comparable; se usó yolo11n como baseline histórico.
- Discrepancia FP 6 vs 35 en `export_ensemble_fp.py`: causada por usar `--cell bin_full --threshold 0.5` vs config low-FP en `bin_filtered` (`49|57 @ 0.86`).
- Mejor modelo solo binario: **modelo_12** (`bin_full`, argmax).
- Mejor ensemble F1: **11|12 mean @ 0.50** (FP=36 en eval original).
- Mejor ensemble low-FP campaña: **11|40 AND @ 0.74** (FP=6 en eval clip completo).
