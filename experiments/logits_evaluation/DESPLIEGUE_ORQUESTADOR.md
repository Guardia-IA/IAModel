# Consenso de 2 modelos para detección de robo — Especificación para el Orquestador

> Objetivo: a partir de las salidas de **dos modelos** (`modelo_54` y `modelo_57`) sobre
> un **chunk de vídeo de 3 segundos**, el orquestador debe calcular **un único valor
> numérico** y decidir **robo / no-robo** con **un único umbral**, minimizando falsos
> positivos sin perder detección de robos.

---

## 1. Resumen ejecutivo (TL;DR)

Por cada chunk, el agente de inferencia devuelve, **para cada modelo**, `logit0`, `logit1` y
`gap = logit1 - logit0`.

El orquestador hace:

```text
score = min(gap_modelo_54, gap_modelo_57)
robo  = score > X        # X recomendado = 2.3
```

- **`min`** → exige que **ambos** modelos estén convencidos (lógica tipo AND) ⇒ menos falsos positivos.
- **`gap`** (no probabilidad/sigmoide) → umbral estable, legible y sin saturación.
- **`X = 2.3`** → punto equilibrado (ver tabla en §5). Subir X = menos falsas alarmas pero menos robos detectados.

---

## 2. Entradas que recibe el orquestador

Para cada chunk de 3 s, el agente de inferencia aplica **los dos modelos** y entrega:

| Campo | Tipo | Descripción |
|---|---|---|
| `chunk_id` | string | Identificador del trozo de vídeo (3 s). |
| `modelo_54.logit0` | float | Logit crudo de la clase 0 (no-robo). |
| `modelo_54.logit1` | float | Logit crudo de la clase 1 (robo). |
| `modelo_54.gap` | float | `logit1 - logit0`. |
| `modelo_57.logit0` | float | Logit crudo de la clase 0 (no-robo). |
| `modelo_57.logit1` | float | Logit crudo de la clase 1 (robo). |
| `modelo_57.gap` | float | `logit1 - logit0`. |
| `valid` | bool | (Opcional) Indica si la pose del chunk es evaluable (ver §6). |

> **Nota:** los modelos devuelven **logits crudos** (la red termina en una capa lineal, sin
> softmax). Por eso trabajamos con `gap = logit1 - logit0` y **no** con probabilidades.

Ejemplo de payload de entrada:

```json
{
  "chunk_id": "cam3_2025-12-15_00:00:00_003",
  "modelo_54": { "logit0": -1.82, "logit1":  3.10, "gap": 4.92 },
  "modelo_57": { "logit0": -0.55, "logit1":  1.40, "gap": 1.95 },
  "valid": true
}
```

---

## 3. Por qué se usa el `gap` y no la probabilidad / sigmoide

1. **Misma decisión, umbral más robusto.** Para un clasificador binario,
   `P(robo) = sigmoide(gap)`. Poner umbral en la probabilidad es **equivalente** a ponerlo
   en el gap (es una transformación monótona): clasifica exactamente igual.
2. **Sin saturación.** El sigmoide aplasta los valores cerca de 0 y 1: gaps de 8 y de 15
   (uno seguro, otro segurísimo) caen ambos en ~0.9997 y dejan de distinguirse. En espacio
   de gap **no se comprimen**, así que el umbral es estable y legible (p. ej. `2.3` o `6.7`
   en vez de `0.998…`, sensible al último decimal).
3. **Alineado con el entrenamiento.** El modelo se entrena con `argmax(logits)`, cuya
   frontera es exactamente `gap > 0`.

---

## 4. Por qué fusionar con `min` (y no media)

- **NO usar media de logits crudos:** `modelo_54` y `modelo_57` están en **escalas
  distintas** (sus umbrales óptimos individuales son ~5.2 y ~7.1). Promediar logits sesga
  hacia el modelo de valores más grandes.
- **`min(gap_54, gap_57)`** = lógica **AND**: solo dispara si **los dos** modelos superan el
  umbral. Si uno duda, no hay alarma ⇒ **caen los falsos positivos**.
- Alternativas (informativo):
  - `mean` → voto equilibrado (más recall, más FP).
  - `max` → lógica OR (más recall, muchos más FP). **No recomendado** aquí.

---

## 5. Umbral X y compromiso falsos positivos / detección

Datos medidos sobre **8.603 ventanas** (1.721 de robo, 6.882 de no-robo), score = `min(gap_54, gap_57)`:

| X (gap) | Acierto | Recall (robos detectados) | FP | FPR | Precisión | Perfil |
|---|---|---|---|---|---|---|
| **0.0** | 95.5% | 89.8% | 210 | 3.05% | 88.0% | Sensible (más alarmas) |
| **2.3** | 95.9% | 87.9% | 142 | 2.06% | 91.4% | **Equilibrado (recomendado)** |
| **6.7** | 95.2% | 79.9% | 68 | 0.99% | 95.3% | Conservador (pocas alarmas) |

Lectura: **subir X ⇒ menos falsos positivos pero se escapan más robos** (baja el recall).

**Recomendación de despliegue:**
- Arrancar con **`X = 2.3`**.
- Si en producción hay demasiadas falsas alarmas, subir X gradualmente hacia `6.7`,
  vigilando que el recall (robos detectados) no baje de lo tolerable.

---

## 6. Casos límite (qué debe hacer el orquestador)

| Situación | Acción recomendada |
|---|---|
| `valid = false` (pose no evaluable: <36 frames válidos, faltan keypoints, etc.) | **No decidir robo**. Marcar el chunk como `no_evaluable` y no contar como no-robo. |
| Falta la salida de uno de los dos modelos | **No disparar alarma** (sin consenso no hay AND). Registrar el incidente. |
| `gap` con `NaN` / `inf` | Tratar como `no_evaluable`. |
| Empate en el límite (`score == X`) | Regla: `robo` solo si `score > X` (estrictamente mayor). |

> Criterio de pose válida (lo aplica el agente de inferencia): el `poses.npy` del chunk debe
> tener **>36 frames** con los **8 keypoints** visibles (no `(0,0)` ni `NaN`).

---

## 7. Pseudocódigo de referencia

```python
X = 2.3  # umbral de despliegue (ajustable; ver §5)

def decidir_robo(salida_chunk) -> dict:
    if not salida_chunk.get("valid", True):
        return {"decision": "no_evaluable", "score": None}

    m54 = salida_chunk.get("modelo_54")
    m57 = salida_chunk.get("modelo_57")
    if m54 is None or m57 is None:
        return {"decision": "no_evaluable", "score": None}

    gap_54 = m54["gap"]  # == m54["logit1"] - m54["logit0"]
    gap_57 = m57["gap"]

    if any(_es_invalido(v) for v in (gap_54, gap_57)):  # NaN / inf
        return {"decision": "no_evaluable", "score": None}

    score = min(gap_54, gap_57)
    return {
        "decision": "robo" if score > X else "no_robo",
        "score": score,
        "umbral": X,
    }
```

Ejemplo de salida:

```json
{ "chunk_id": "cam3_...", "decision": "no_robo", "score": 1.95, "umbral": 2.3 }
```

---

## 8. Recomendaciones operativas

- **Monitorización:** registrar `score`, `decision` y, si se confirma a posteriori,
  el resultado real (robo/no-robo) para recalcular FP/recall periódicamente y reajustar X.
- **Reajuste de X:** es el único parámetro que se toca en producción. No requiere
  reentrenar modelos.
- **Persistencia temporal (mejora futura, no incluida):** si se desea reducir aún más los
  FP, se puede exigir que **N chunks consecutivos** superen X antes de declarar robo. Esto
  reduce alarmas esporádicas a costa de unos segundos de latencia. (Pendiente de validar.)
- **Modelos:** `modelo_54` y `modelo_57` fueron seleccionados como la mejor pareja para
  consenso (mejor recall a igual nivel de FP que cualquier modelo individual).

---

## 9. Parámetros (resumen para configuración)

```yaml
consenso:
  modelos: [modelo_54, modelo_57]
  valor_por_modelo: gap          # gap = logit1 - logit0
  fusion: min                    # AND-like, minimiza falsos positivos
  umbral_X: 2.3                  # equilibrado; conservador = 6.7
  regla: "robo si score > umbral_X"
```
