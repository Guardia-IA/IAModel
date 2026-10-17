# Brief para experto — Ventanas temporales (train/eval vs producción) y arquitecturas

**Fecha:** 2026-10-17  
**Idioma:** español (documento para revisión externa / agente experto).  
**Versión en inglés 100% autocontenida (pasar solo este fichero al agente experto):** [`CAMPAIGN2_EXPERT_BRIEF_WINDOWS_AND_ARCHITECTURES_EN.md`](CAMPAIGN2_EXPERT_BRIEF_WINDOWS_AND_ARCHITECTURES_EN.md)  
**Proyecto:** Kanvis — detección de hurtos por esqueleto (Stage 1).  
**Repositorio:** `IAModel/experiments/training/` (`campaign/` legacy, `campaign2/` + `train_v2/` nuevo).

---

## 1. Objetivo de este documento

Pedimos **valoración experta** sobre dos cuestiones concretas:

1. **Contrato temporal:** entrenamiento y validación con **archivos `.npy` completos** (duración variable por clip) frente a **producción**, donde el modelo recibe **trozos fijos** (~8 s, configurable). ¿Conviene **evaluar** y/o **entrenar** con ventanas de **2, 3, 4, 6 u 8 segundos** (cuando el clip lo permita) para elegir el marco en producción?
2. **Arquitecturas:** más allá de la matriz Campaign2 actual (TCN enmascarado + GRU pooled), ¿qué familias recomendaríais para **~19k clips etiquetados** (~2,6k cat. 6 robo/ocultación), sin repetir un grid de **60 experimentos**?

Incluimos contexto, lo implementado, resultados preliminares y un ejemplo de `.npy` tipo producción.

---

## 2. Contexto de negocio y datos

| Tema | Detalle |
|------|---------|
| **Problema** | Clasificar secuencias de pose (8 joints superiores) para alarma de robo/ocultación en supermercado. |
| **Prioridad owner** | **FP primero** (≈ ≤1 FP/día objetivo aspiracional; presupuesto de trabajo más realista ~10 FP/1000 sesiones en desarrollo). |
| **Datos etiquetados (oct 2026)** | ~**19 043** clips, ~**2 592** cat. **6** (acción etiquetada por humano en segmento mp4→npy). Una tienda, **9 cámaras**. |
| **Etiqueta** | Una categoría por clip; subclips con etiqueta propia. Cat. 6 mezcla gestos de ocultación; parte son **falsos “robo” de negocio** (p. ej. guardar el móvil). |
| **Legacy val (referencia)** | Mejor ensemble **#41+#48 AND @ 0.80** → F1 ~76 %, recall ~64 %, **11 FP / 1 619 negativos** (~0,68 % clip-level). |
| **Producción (referencia)** | Orden de **~20–30 FP / ~650–1000 sesiones** (unidades distintas a val clip-level; ver auditoría). |

**Documentos de referencia en repo:**

- `IAModel/docs/TRAINING_AUDIT.md`
- `IAModel/docs/TRAINING_EXPERT_RESPONSE_20261015.md`
- `IAModel/docs/TRAINING2_CAMPAIGN2_MASTER_PLAN.md`
- `IAModel/docs/KANVIS_POSE_SHARED_PROJECT_SPEC.md` (paridad pose train ↔ C1)

---

## 3. Desajuste histórico: clip entero vs producción

### 3.1 Entrenamiento legacy (`campaign/` + `train_model_operations.py`)

- Input: **`.npy` por clip** (duración **variable**: acciones cortas o largas).
- Preprocesado típico: normalización global → velocidad → **`temporal_resize`** al `seq_len` del modelo (p. ej. 64–128 pasos).
- Efecto: **comprimir/estirar** temporalmente todo el clip en una secuencia fija → **tiempo físico distorsionado** (auditoría H4).
- Evaluación: a menudo **nivel clip** (un score por archivo), split por clip, umbral a veces optimizado en val.

### 3.2 Producción (C1 / kanvis-monitoring)

- Sesiones por persona/track; emisión de **chunks `.npy` de ventana fija**.
- Spec compartido (§5.4 `KANVIS_POSE_SHARED_PROJECT_SPEC.md`): **`window_seconds = 8`**, **`stride_seconds = 4`** por sesión; salida `poses_full.npy`, `valid_mask.npy`, etc.
- Inferencia: el modelo ve **fragmentos de ~8 s** (ajustable), no necesariamente el clip de entrenamiento completo.
- Además existe desajuste **extractor offline (Ultralytics entrenamiento) vs Triton/TensorRT producción** (proyecto aparte de paridad de pose).

### 3.3 Hipótesis del owner (pregunta actual)

> “Entreno con npy completos de distinta duración, pero en producción uso trozos de 8 s. ¿Puede explicar calidad? ¿Deberíamos evaluar/entrenar con trozos de 2–8 s?”

La auditoría y la respuesta experta 2026-10-15 ya señalan este **mismatch clip-centrado vs ventanas de replay** como causa **prioritaria** de FP en producción (orden de magnitud comparable a etiquetas/protocolo).

---

## 4. Qué hemos implementado (Campaign2 + train_v2)

Pipeline **paralelo en GPU** (RTX 5090, hasta ~8 jobs): `IAModel/experiments/training/campaign2/`.

### 4.1 Contrato temporal v2 (alineación parcial con producción)

| Parámetro | Valor actual (`campaign2_config.json`) |
|-----------|----------------------------------------|
| `target_hz` | **12 Hz** (remuestreo desde fps en `meta.json`) |
| `window_seconds` | **8.0 s** → **96 frames** (`seq_len`) |
| `stride_seconds` | **1.0 s** entre ventanas (train/eval offline) |
| Almacenamiento | Caché mmap contigua por clip completo remuestreado; **ventanas deslizantes** sobre ese array (no se recortan físicamente archivos `.npy` en disco). |
| Máscaras | NaN → features con flags; padding al final de clip con máscara (no borrar frames del tensor). |
| Normalización | Torso, coords isotrópicas `(x×W/H, y)`, ~62 canales (posición, vel, validez, distancias muñeca–torso). |
| Etiquetas ventana | Política **`tail_positive`**: solo últimos ~3 s positivos en clips cat. 6; MIL bag LSE auxiliar. |
| Selección checkpoint | **recall@FP** en val agrupado (presupuesto ~11 FP clips en val). |
| Umbral | Calibración en split **T** (`threshold_calib`), order-statistic en negativos; eval en **val_dev**. |
| Ensemble v2 | Media de probabilidades entre **3 semillas** (17, 42, 103); legacy #41+#48 solo **baseline P0**. |

### 4.2 Matriz de entrenamiento actual

- **2 arquitecturas:** `masked_tcn64` (TCN dilated width 64, readout mean+max, cabeza concealment), `gru64_pooled`.
- **3 semillas** × **2 pose sources** (`poses_full.npy` vs `poses.npy` filtered) → **12 entrenamientos** por campaña matriz.
- **100 épocas** (debug: 4 épocas, ~50 clips).

### 4.3 Flujo operativo

```text
probe → preflight (--matrix, inventario + caché full/filtered) → train paralelo
→ p1-calibrate → p1-eval → resúmenes / ensemble-leaderboard
```

Comando producción: `./run_campaign2.sh matrix-pipeline --data-root /media/8TB/.../data_clips/`

### 4.4 Lo que **aún no** replica producción al 100 %

- **Stride 1 s** en train vs **stride 4 s** en producción (spec C1).
- **Agregación sesión** (max sobre ventanas + política de cooldown/dedup) limitada en eval offline (clip-max).
- Clips de entrenamiento siguen siendo **segmentos “acción completa”** etiquetados, no ventanas aleatorias fuera de segmento (background mining).
- Paridad pose offline/producción en curso (spec Kanvis).

---

## 5. Resultados preliminares (no conclusivos)

### 5.1 Debug-matrix (2026-10-17, ~50 clips, 4 épocas)

- Pipeline **end-to-end OK** tras fixes (imports, manifest matriz, inventario debug acotado).
- **P1 eval** en `val_dev` con **n≈5** clips: **Recall ~0 %**, **FP 0–1** — **esperable** en debug (pocos robos en split, entrenamiento mínimo).
- Objetivo del run: validar **infra**, no métricas de producto.

### 5.2 Estimaciones de tiempo (dataset completo ~19k clips)

- Heurística con factor calibración ×7: **varias horas** de pared en GPU con 8 jobs para P1 matriz (preflight + 12 trains), según ventanas/época reales del manifest.
- Run **real** pendiente de lanzar en wudebian con `--data-root` completo.

### 5.3 Posición del experto previo (2026-10-15)

- Recomendación principal: **8 s @ 12 Hz, 96 pasos**, stride **1 s** en desarrollo; comparar **4 s y 12 s** después del baseline, **no** barrer docenas de longitudes sin protocolo.
- **No** priorizar sweep de 60 arquitecturas; primero contrato temporal + eval agrupada.
- Alternativas compactas **después** del TCN: **GRU con pooling enmascarado correcto**, **ST-GCN anatómico**, **Transformer pequeño** con posición.

---

## 6. Ejemplo adjunto: `mergedfps.npy` (tipo producción)

Archivo de ejemplo proporcionado por el owner: `/home/debian/Descargas/mergedfps.npy`

| Campo | Valor |
|-------|--------|
| `shape` | **(60, 8, 2)** — float32 |
| Interpretación | **60 frames** × **8 joints** × **(x, y)** normalizados |
| Duración @ 12 Hz | **5,0 s** (no 8 s) |
| Rango coords | ~0,10 – 0,49 (espacio normalizado imagen) |

**Nota:** Es un **fragmento corto** respecto a la ventana de 8 s / 96 frames del entrenamiento v2 actual. En producción, fragmentos finales de sesión pueden ser **< 8 s** (padding/máscara en v2); en legacy, `temporal_resize` los estiraba/comprimía de otra forma.

Este archivo ilustra que **producción no siempre entrega 8 s llenos**, lo cual interactúa con la pregunta de longitudes 2–8 s.

---

## 7. Pregunta 1 — Ventanas de 2, 3, 4, 6 u 8 s en entrenamiento y/o evaluación

### 7.1 Planteamiento del owner

1. **Evaluación:** ¿Entrenar **una vez** (p. ej. con ventana 8 s) y **evaluar** el mismo checkpoint pasando trozos de **2, 3, 4, 6, 8 s** (recortando/padding con máscara) para medir en qué marco temporal rinde mejor y **fijar producción**?
2. **Entrenamiento:** ¿Es deseable **entrenar** (o fine-tune) con trozos de **2–8 s** muestreados de los `.npy` completos (multi-scale training), no solo ventanas de 8 s?

### 7.2 Lo que ya hace Campaign2 (para acotar la pregunta)

- **No** entrena con “un npy entero resizeado a seq_len” (salvo rama P0 legacy).
- **Sí** entrena con **ventanas deslizantes de 8 s** sobre clips remuestreados a 12 Hz; clips cortos → menos ventanas o ventana parcial con pad.
- **No** hay yet grid automático 2/3/4/6/8 s ni alineación de stride con producción (4 s).

### 7.3 Subpreguntas para el experto

| # | Pregunta |
|---|----------|
| A | ¿El mismatch **clip completo legacy** vs **chunk 8 s prod** sigue siendo relevante **después** de pasar a ventanas 8 s/12 Hz en train_v2, o el problema pasa a ser sobre todo **stride**, **agregación sesión** y **distribución de fragmentos** (como `mergedfps.npy` 5 s)? |
| B | ¿Recomendáis **eval multi-longitud** (mismo pesos, distinto `window_seconds` / `seq_len`) en un **replay agrupado** antes de tocar train? ¿Orden de longitudes a probar (p. ej. 4, 6, 8, 12 s como dijo el experto previo vs grid 2–8 s del owner)? |
| C | Si se evalúa multi-longitud: ¿**mismo `seq_len` fijo** con muestreo temporal, o **seq_len proporcional** (2 s→24, 8 s→96)? ¿Cómo evitar confundir efecto de escala temporal con efecto de información contenida? |
| D | ¿**Entrenamiento multi-scale** (2–8 s) aporta robustez real en FP/recall sesión, o aumenta riesgo de **etiquetas débiles** (ventanas positivas fuera del gesto) salvo intervalos anotados? |
| E | ¿Estrategia **4 s “evidencia” + 8 s “contexto”** (multi-instantánea / two-stream) vs **single-scale** 8 s para hurtos breves? |
| F | ¿Cómo alinear experimentos con producción: entrenar stride 1 s pero desplegar stride 4 s — ¿eval obligatoria con **stride de producción** y **max/mean sobre ventanas solapadas**? |
| G | Para clips etiquetados “acción completa”: ¿recortar **solo ventanas dentro del segmento** vs ventanas en todo el npy (incl. approach/walk-up) cambia el veredicto sobre multi-longitud? |
| H | Criterio de decisión: ¿métrica primaria **recall evento @ FP/sesión** por longitud, con **paired bootstrap** entre longitudes en el mismo split agrupado? |

### 7.4 Propuesta experimental mínima (para que el experto la valide o corrija)

**Fase E1 — Solo eval (barato):**

- Checkpoint: TCN v2 entrenado 8 s/96 (3 semillas).
- Replay: mismo split agrupado, mismas sesiones.
- Longitudes: **4, 6, 8** s (opcional 3 s si hay robos muy cortos); **seq_len = 12 × segundos**.
- Agregación: **max** por clip/sesión (como prod inicial); repetir con stride **1 s y 4 s**.
- Salida: curva recall@FP vs longitud; recomendación de `window_seconds` prod.

**Fase E2 — Train (caro, solo si E1 lo justifica):**

- Opción 2a: entrenar **solo** la longitud ganadora en E1.
- Opción 2b: entrenar **multi-scale** (muestreo uniforme de {4,6,8} s) con misma arquitectura (ajuste de `seq_len` por batch o batch por escala).
- Opción 2c: modelo **two-tower** 4 s + 8 s (experto § evidencia/contexto).

Pedimos al experto: **¿E1/E2 es sensato? ¿Qué falta?**

---

## 8. Pregunta 2 — Arquitecturas recomendables (shortlist, no grid de 60)

### 8.1 Estado actual

- Implementadas y en cola: **Masked TCN 64**, **BiGRU pooled 64** (~0,05–0,14 M params).
- **No** implementadas en `train_v2`: ST-GCN anatómico, Transformer pequeño, #48 legacy con loader v2, baselines lineales.
- Legacy probó **~20 ids** de `model_config.EXPERIMENTS` bajo protocolo cuestionado (H1/H2/H3).

### 8.2 Restricciones del owner

- **GPU:** RTX 5090; preferencia por modelos **ligeros** si métricas similares (≤2 pp recall @ mismo FP, regla Pareto).
- **Datos:** ~19k clips pero **pocos robos independientes a nivel sesión/evento**; muchos negativos; etiqueta ruidosa en cat. 6.
- **No** repetir campaña de **60 arquitecturas**; buscar **10–20 candidatos máximo** bien justificados, o **4–6 familias × pocos hparams × 3 semillas**.

### 8.3 Candidatos ya citados por auditoría / experto

| Familia | Motivo | Riesgo / nota |
|---------|--------|----------------|
| **Masked dilated TCN** (width 64) | Receta principal experto; RF ~125 @ 12 Hz; enmascarado | Ya en producción v2 |
| **BiGRU + masked mean/max** | Corrige pooling legacy; muy ligero | En cola Campaign2 |
| **ST-GCN / CTR-GCN lite** (grafo **anatómico** 8 joints) | Estructura espacial; comparador | Legacy ST-GCN con grafo incorrecto |
| **Transformer / Conformer pequeño** (d=64, PE + mask) | Solo si empata TCN/GRU | Sin PE → orden ambiguo (auditoría M1) |
| **TCN-LSTM / TCN-GRU reparado** (#41/#48) | Comparador incumbent | Pesado (~1–3 M); AND legacy no es stack v2 |
| **Linear / GBDT** sobre features hand-crafted | Techo de información en features | Barato; útil diagnóstico |
| **Seed ensemble** (3× misma arq.) | Estabilidad vs diversidad arq. | Recomendado v2 vs AND #41+#48 |

### 8.4 Subpreguntas para el experto

| # | Pregunta |
|---|----------|
| A | Con **~2,6k clips positivos** pero **pocos eventos/sesiones independientes**, ¿cuánta capacidad (params) es prudente? ¿TCN 0,14M vs GRU 0,08M vs ST-GCN 0,1–0,4M? |
| B | ¿Shortlist **final de 4–6 familias** (no 60) que probaríais en **P2** tras baseline 8 s, con **3 semillas** cada una? |
| C | ¿Tiene sentido **re-entrenar #48** con loader v2 (misma seq 96) como puente legacy→v2, o solo distrae? |
| D | ¿**Ensemble** recomendado: media 3 semillas TCN, o TCN+GRU, o mantener #41+#48 para prod mientras v2 madura? |
| E | ¿Alguna arquitectura **skeleton SOTA** (PoseC3D, MS-G3D, etc.) merece el coste aquí, o el cuello de botella es **datos/protocolo**? |
| F | Dado el ejemplo **`mergedfps.npy` (60 frames, 5 s)**, ¿cambiaría la elección de arquitectura (p. ej. más sensibilidad a **secuencias cortas** → GRU vs TCN)? |

### 8.5 Datos de entrada (recordatorio para diseño de arq.)

- **8 joints:** hombros, codos, muñecas, caderas (orden legacy).
- Canales v2: ~**62** (pos, vel, validez, distancias, flags anchor/scale).
- **Máscaras** obligatorias en forward; abstención si input inválido (objetivo experto; parcialmente implementado).

---

## 9. Información que el experto puede solicitar

- Manifest y splits de un run real (post `matrix-pipeline` completo).
- `reports/p0_baseline.json`, `ensemble_leaderboard.json`, curvas recall@FP por job.
- Muestras de duración de clips (distribución de frames @ 12 Hz post-caché).
- Logs de paridad pose (cuando existan) train vs C1.
- Lista de categorías y conteos por split.

---

## 10. Resumen ejecutivo de las dos cuestiones

1. **Ventanas 2–8 s:** Campaign2 **ya acerca** train a producción con **ventanas 8 s @ 12 Hz** sobre npy completos remuestreados, pero **no** replica stride prod (4 s), ni estudia longitudes alternativas. El owner pregunta si conviene un **estudio sistemático eval-only y/o train multi-scale** para elegir duración en producción — especialmente relevante para fragmentos **< 8 s** como el ejemplo `mergedfps.npy`.

2. **Arquitecturas:** La propuesta original al agente **no** pedía otro grid de 60 modelos; pedía **TCN pequeño enmascarado + 3 semillas** y comparadores **GRU/ST-GCN/Transformer lite**. Campaign2 implementa **TCN + GRU**. Pedimos **shortlist priorizada (≈4–10 candidatos)** para la siguiente fase, acorde a volumen y sesgo de los datos actuales.

---

## 11. Referencias cruzadas

| Documento | Contenido relevante |
|-----------|---------------------|
| `TRAINING_EXPERT_RESPONSE_20261015.md` §4.3–4.4 | 8 s/12 Hz, ventanas, TCN width 64, comparar 4/12 s después |
| `TRAINING_AUDIT.md` H4, §16 | `temporal_resize`, padding, ventanas producción |
| `TRAINING2_CAMPAIGN2_MASTER_PLAN.md` | Fases P0–P2, matriz reducida |
| `KANVIS_POSE_SHARED_PROJECT_SPEC.md` §5.4 | Chunks 8 s, stride 4 s en C1 |
| `campaign2/README.md` | Comandos y estado Sprint 3 |

---

*Documento generado para revisión externa. No incluye veredicto propio sobre longitud óptima ni ranking arquitectural definitivo — se delega al experto.*
