# Handoff: inferencia ensemble robo (modelo_06 + modelo_14)

Documento para **otro agente** que implemente la inferencia en código.

**Objetivo:** cargar dos checkpoints binarios de la campaña, calcular P(robo) con cada uno, combinar con **media (MEAN)** y decidir alarma con umbral **0.68**.

**No tunear** umbral ni regla en test; están fijados en val.

---

## Resumen en una frase

```
alarma_robo = ( P_robo(modelo_06) + P_robo(modelo_14) ) / 2  >=  0.68
```

---

## ⚠️ Aclaración importante: NO es argmax del ensemble

| Paso | Qué hacer | Qué NO hacer |
|------|-----------|--------------|
| Por cada modelo | `softmax(logits)[1]` → **P(robo)** | No usar `argmax(logits)` como decisión final |
| Combinación | **Media** de las dos probabilidades | No hacer argmax por modelo y votar |
| Decisión | `P_mean >= 0.68` | No usar umbral 0.8 por defecto de `test_model.py` |

Cada modelo **internamente** produce logits → softmax; la **decisión del sistema** es umbral sobre la **media**, no argmax.

---

## Configuración fija (producción / eval test)

| Campo | Valor |
|-------|--------|
| Modo | **Binario** (2 clases: 0=no robo, 1=robo) |
| Celda entrenamiento | `bin_full_hardened` |
| Pose | **`full`** → `poses_full.npy` (+ `valid_mask.npy` si existe) |
| `single_user_only` | `true` (como en campaña) |
| Modelo A | **Experimento 06** → `modelo_06.pt` |
| Modelo B | **Experimento 14** → `modelo_14.pt` |
| Regla ensemble | **`mean`** |
| Umbral | **`0.68`** |
| Split plan | `training_plan.json` de la misma celda |

### Métricas de referencia (split val — sanity check)

Tras implementar, en **val** deberías acercarte a:

- F1 ≈ **73,3%**
- Recall ≈ **61,8%**
- FP rate ≈ **1,94%** (13 FP sobre 671 negativos)

Si en val sale muy distinto, revisar preprocessing o índice de clase positiva.

---

## Rutas de ficheros

### Checkpoints (obligatorio: ambos de la misma celda)

Relativas al repo:

```
experiments/training/campaign/artifacts/models/bin_full_hardened/modelo_06.pt
experiments/training/campaign/artifacts/models/bin_full_hardened/modelo_14.pt
```

En servidor de entrenamiento (Angel):

```
/home/angel/IAModel/experiments/training/campaign/artifacts/models/bin_full_hardened/modelo_06.pt
/home/angel/IAModel/experiments/training/campaign/artifacts/models/bin_full_hardened/modelo_14.pt
```

### Plan de split y datos

```
experiments/training/campaign/artifacts/plans/bin_full_hardened/training_plan.json
```

`data_root` dentro del plan (o `GUADIA_DATA_RESULT_ROOT` / `--data-root`).

---

## Arquitecturas (leer del checkpoint, no hardcodear a ciegas)

| Fichero | Exp | `arch` | `seq_len` | Notas |
|---------|-----|--------|-----------|--------|
| `modelo_06.pt` | 06 | `tcn` | **96** | hidden 256, dropout 0.3 |
| `modelo_14.pt` | 14 | `res_tcn` | **64** | hidden 256, num_blocks 3 |

Al cargar cada `.pt`:

```python
ckpt = torch.load(path, map_location=device, weights_only=False)
cfg = ckpt["config"]
arch = cfg["arch"]
seq_len = int(ckpt.get("seq_len", 64))
input_dim = int(ckpt["input_dim"])
num_classes = int(ckpt.get("num_classes", 2))  # debe ser 2
task = ckpt.get("task", "binary")
positive_class = int(ckpt.get("positive_class", 6))  # metadata; label binaria positiva = índice 1

model = build_model(arch, input_dim, num_classes, cfg)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

**Cada modelo usa su propio `seq_len`** en `temporal_resize`.

---

## Pipeline de preprocessing (igual que entrenamiento/eval)

Reutilizar funciones de `train_model_operations.py`:

1. Cargar poses: `poses_full.npy` desde `data_result/{cat}/{clip}/user_X/`
2. Si hay `valid_mask.npy`, filtrar frames inválidos (NaN)
3. `normalize_sequence(poses)`
4. `add_velocity(poses)`
5. `temporal_resize(poses, seq_len)` — **seq_len distinto por modelo**
6. Reshape a `[T, features]` → tensor `[1, T, features]`

Referencia de eval batch: `evaluate_campaign.collect_binary_predictions` y `build_pose_dataset_for_eval`.

Para **un clip**:

```python
import torch.nn.functional as F

@torch.no_grad()
def prob_robo(model, x_tensor, device) -> float:
    x = x_tensor.to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    return float(probs[0, 1].item())  # clase 1 = robo en binario
```

---

## Regla ensemble (implementación)

```python
THRESHOLD = 0.68

p06 = forward_prob_robo(model_06, poses, seq_len=96, device)
p14 = forward_prob_robo(model_14, poses, seq_len=64, device)

p_mean = (p06 + p14) / 2.0
alarma = p_mean >= THRESHOLD

# Salida recomendada por clip
result = {
    "p_modelo_06": p06,
    "p_modelo_14": p14,
    "p_mean": p_mean,
    "threshold": THRESHOLD,
    "alarma_robo": alarma,
}
```

---

## Qué debe implementar el agente

### Opción A — Script batch sobre `data_result` (prioritario)

Crear p. ej. `experiments/training/campaign/infer_ensemble_bin_full_hardened.py` que:

1. Cargue los **dos** `.pt` de `bin_full_hardened`.
2. Lea ejemplos con `collect_examples(pose_source="full", single_user_only=True)` + split del plan (`val` o `test`).
3. Por cada clip, calcule `p06`, `p14`, `p_mean`, `alarma`.
4. Imprima métricas binarias (TP/FP/FN/TN, F1, recall, FP rate).
5. Guarde CSV con columnas: `uid`, `folder_category`, `p_modelo_06`, `p_modelo_14`, `p_mean`, `alarma`, `clip_video_path`, etc.
6. CLI mínima:

```bash
cd experiments/training/campaign
python infer_ensemble_bin_full_hardened.py --split test
python infer_ensemble_bin_full_hardened.py --split val   # sanity check ~1.94% FP
```

**Reutilizar** sin duplicar lógica:

- `export_ensemble_fp.run_ensemble_export` (ya existe) — eval + CSV FP
- `train_model_operations.build_model`, `collect_examples`, `build_split_examples` vía `evaluate_validation`

Si solo hace falta eval + CSV de FP, puede extender `export_ensemble_fp.py` con `--split test` (ya soportado).

### Opción B — Integración en vídeo en vivo (`test_model.py`)

Hoy `test_model.py` usa **un solo** `--model` y umbral default **0.8**.

Cambios sugeridos:

1. Añadir flags opcionales:
   - `--model-b` (segundo checkpoint)
   - `--ensemble-rule mean` (default `mean` si hay dos modelos)
   - `--threshold-robbery 0.68` (default cuando ensemble activo)
2. En el loop de inferencia por track:
   - Calcular `p_a`, `p_b` con el **mismo** preprocessing pero **distinto seq_len** por checkpoint
   - `p = (p_a + p_b) / 2.0`
   - `is_robber = p >= threshold`
3. Mostrar overlay: `P06`, `P14`, `P_mean`.

**No** romper el modo single-model existente.

### Opción C — Módulo reutilizable

Crear `experiments/training/campaign/ensemble_infer.py` con:

```python
class RobberyEnsemble:
    def __init__(self, model_paths: list[Path], rule: str = "mean", threshold: float = 0.68): ...
    def load(self, device): ...
    def predict_proba(self, poses: np.ndarray) -> dict: ...  # p06, p14, p_mean
    def predict_alarm(self, poses: np.ndarray) -> bool: ...
```

Usado por batch script y por `test_model.py`.

---

## Código existente útil (no reinventar)

| Fichero | Uso |
|---------|-----|
| `campaign/export_ensemble_fp.py` | **Ya implementa** eval MEAN 06+14 @ 0.68 + CSV FP |
| `campaign/evaluate_campaign.py` | `collect_binary_predictions`, métricas |
| `evaluate_validation.py` | Split, `build_split_examples` |
| `train_model_operations.py` | `build_model`, preprocessing, `collect_examples` |
| `post_training_binary_tools.py` | Referencia forward P(robo) por clip |
| `test_model.py` | Integración vídeo + YOLO (opcional) |

---

## Checklist de verificación

- [ ] Dos modelos cargados desde **`bin_full_hardened`** (no mezclar celdas)
- [ ] `task=binary`, `num_classes=2`, probabilidad clase **índice 1**
- [ ] `poses_full.npy` (no `poses.npy`)
- [ ] `seq_len` **96** para 06, **64** para 14 (desde checkpoint)
- [ ] Decisión = **media de probabilidades**, umbral **0.68**
- [ ] Sanity check en **val**: FP ~1,9%, recall ~62%
- [ ] Eval final en **test** sin cambiar umbral

---

## Alternativas documentadas (NO implementar salvo petición explícita)

| Config | Regla | F1 val | Recall | FP | Alarmas/día* |
|--------|-------|--------|--------|-----|--------------|
| **Recomendada (esta)** | MEAN 06+14 @ 0.68 | 73,3% | 61,8% | 1,94% | ~19 |
| Más recall | AND 06+17 @ 0.50 | 76,2% | 68,1% | 2,98% | ~30 |
| Más recall | AND 06+12+17 @ 0.50 | 75,8% | 65,4% | 2,09% | ~21 |
| Un solo modelo (no usar) | argmax 06 | 77,9% | 78,5% | 6,56% | ~66 |

\* 1000 interacciones/día.

---

## Comandos rápidos post-implementación

```bash
cd experiments/training/campaign

# Eval + lista FP en val (sanity check)
python export_ensemble_fp.py --split val

# Eval en test (una sola pasada)
python export_ensemble_fp.py --split test --outcomes all

# Export FP con rutas vídeo completas
python export_ensemble_fp.py --split val --export-videos
```

---

## Referencia cruzada

- Config detallada: `RECOMMENDED_ENSEMBLE_BIN_FULL_HARDENED.md`
- Resultados campaña: `artifacts/reports/_master/campaign_ensemble_val.csv`
- Fila exacta: `bin_full_hardened`, `ensemble_mean`, `modelo_06|modelo_14`, thresholds `0.680|0.680`

---

## Petición explícita al agente

1. Implementar **Opción A** (script batch) y/o **Opción C** (módulo `ensemble_infer.py`).
2. Opcional: integrar en `test_model.py` (**Opción B**) para vídeo con overlay `P06 / P14 / P_mean`.
3. Verificar en **val** que las métricas coinciden ~con la tabla de sanity check.
4. Ejecutar **test** una vez y guardar CSV de resultados.
5. **No** cambiar umbral 0.68 ni mezclar modelos de otras carpetas (`bin_full`, `bin_filtered`, etc.).
