# Handoff C2 — Despliegue stage-1 robo (MEAN 49 + 53 @ 0,50)

Documento para el **agente de inferencia C2** (`c2_inference_service` / Triton motion).

**Objetivo:** desplegar el ensemble acordado con IAModel (campaña `campaign_20261103_230735`, celda **`bin_full`**) y alinear paridad preprocess + scoring con entrenamiento.

---

## Resumen en una frase

```text
robbery_score = ( P_robo(modelo_49) + P_robo(modelo_53) ) / 2
alarma_robo     = robbery_score >= 0.50
```

Regla: **`mean`** (no `and`). Umbral: **`0.50`** (no 0,80 ni 0,90).

---

## 1. Configuración de producción

| Campo | Valor |
|--------|--------|
| **Campaña** | `campaign_20261103_230735` |
| **Celda** | **`bin_full`** (NO `bin_full_hardened`, NO `bin_filtered`) |
| **Tarea** | Binario: 2 clases — índice **0** = no robo, índice **1** = robo |
| **Positivo en datos** | Categorías **6** y **14** (en inferencia solo importa el binario) |
| **Entrada pose** | **`poses_full.npy`** + **`valid_mask.npy`** si existe en el chunk |
| **Slot 1** (`p1` / `model.pt`) | **`modelo_49.pt`** |
| **Slot 2** (`p2` / `model2.pt`) | **`modelo_53.pt`** |
| **Regla ensemble** | **`mean`** |
| **Umbral** | **`0.50`** |
| **Sanity val (referencia)** | F1 ~**88,2 %**, recall ~**85,2 %**, ~37 FP / 1909 negativos |

Sustituye el piloto anterior **41 + 48 AND @ 0,80** (otra campaña/celda y regla distinta).

---

## 2. Arquitecturas

Leer hiperparámetros del **checkpoint** (`ckpt["config"]`); no basar el grafo solo en el número de experimento.

### Exp 49 — `tcn_gru` (`PoseTCNGRUClassifier`)

| Parámetro | Valor |
|-----------|--------|
| `arch` | `tcn_gru` |
| `tcn_hidden_dim` | 128 |
| `tcn_layers` | **3** |
| `gru_hidden_dim` | 128 |
| `gru_layers` | 1 |
| `dropout` | 0.25 (solo entrenamiento; inferencia con `eval()`) |
| **`seq_len`** | **96** |

Stack: TCN dilatado + **GRU** + cabeza softmax 2 vías.

### Exp 53 — `conformer_lite` (`PoseConformerLiteClassifier`)

| Parámetro | Valor |
|-----------|--------|
| `arch` | `conformer_lite` |
| `d_model` | 128 |
| `nhead` | 4 |
| `num_layers` | **3** |
| `dim_feedforward` | **384** |
| `conv_kernel` | **9** |
| `dropout` | 0.2 |
| **`seq_len`** | **96** |

Conformer ligero (atención + conv temporal). **No** es el mismo grafo que `tcn_gru`.

Aunque ambos usan `seq_len=96`, hay que **instanciar dos arquitecturas distintas** y cargar cada `model_state_dict` con su `config`.

Definición en catálogo: `experiments/training/model_config.py` → `EXPERIMENTS[48]` (exp 49) y `EXPERIMENTS[52]` (exp 53).

---

## 3. Rutas de checkpoints

**Servidor entrenamiento (referencia artefactos):**

```text
/home/angel/IAModel/experiments/training/campaign/artifacts/runs/campaign_20261103_230735/models/bin_full/modelo_49.pt
/home/angel/IAModel/experiments/training/campaign/artifacts/runs/campaign_20261103_230735/models/bin_full/modelo_53.pt
```

**En máquina Kanvis / C2** (ajustar prefijo):

```text
…/artifacts/runs/campaign_20261103_230735/models/bin_full/modelo_49.pt
…/artifacts/runs/campaign_20261103_230735/models/bin_full/modelo_53.pt
```

Triton: **dos modelos motion** (o dos backends derivados del mismo `.pt`). Documentar `run_id`, `exp_id`, `cell_id=bin_full` en manifest.

---

## 4. Carga del modelo (referencia IAModel)

Implementación: `experiments/training/train_model_operations.py` → `build_model(arch, input_dim, num_classes, cfg)`.

```python
import torch
import torch.nn.functional as F

ckpt = torch.load(path, map_location=device, weights_only=False)
cfg = dict(ckpt["config"])
arch = cfg["arch"]
input_dim = int(ckpt["input_dim"])
num_classes = int(ckpt.get("num_classes", 2))  # debe ser 2
seq_len = int(ckpt.get("seq_len", 64))         # 96 para ambos en esta campaña

model = build_model(arch, input_dim, num_classes, cfg)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Forward
logits = model(x)  # x: [1, T, features] tras preprocess
probs = F.softmax(logits, dim=1)
p_robo = float(probs[0, 1].item())  # índice 1 = robo
```

### Qué NO hacer

| Incorrecto | Correcto |
|------------|----------|
| `argmax(logits)` como decisión final | Umbral sobre **probabilidad** (o media de probabilidades) |
| `robbery_score = min(p1, p2)` (AND) | `robbery_score = (p1 + p2) / 2` con regla **mean** |
| Checkpoints de `bin_full_hardened` | Solo **`bin_full`** de esta campaña |
| `poses.npy` (filtered) sin acuerdo | **`poses_full.npy`** (+ máscara como en train) |

---

## 5. Preprocess (paridad obligatoria)

Misma cadena que entrenamiento / eval / `score_pose_ensemble.py`:

1. Cargar `poses_full` → `[T, J, 2]`.
2. Si existe `valid_mask.npy`, **filtrar** filas con máscara válida (como en train cuando hay máscara).
3. NaN → 0.
4. `normalize_sequence(poses)`.
5. `add_velocity(poses)`.
6. `temporal_resize(poses, seq_len)` con **`seq_len` del checkpoint** (96).
7. Tensor `[1, T, features]` para el modelo.

Un forward **por slot** sobre el **mismo** buffer de poses (misma política de máscara).

---

## 6. Fusión y salida `p1.json`

```python
C2_MOTION_ENSEMBLE_RULE = "mean"
C2_MOTION_ENSEMBLE_THRESHOLD = 0.50

p49 = forward_p_robo(model_49, poses, device)
p53 = forward_p_robo(model_53, poses, device)
robbery_score = (p49 + p53) / 2.0
robbery_alarm = robbery_score >= 0.50
motion_theft_candidate = robbery_alarm
```

Campos recomendados por ventana:

- `p1`: `logit0`, `logit1`, `p_robo` (modelo 49), meta `arch=tcn_gru`, `seq_len=96`
- `p2`: igual (modelo 53), `arch=conformer_lite`
- `robbery_score`: **media** (no `min`)
- `ensemble_rule`: `"mean"`, `ensemble_threshold`: `0.50`

**C3** no debe reaplicar umbral 0,80/0,90 sobre motion salvo política producto explícita.

---

## 7. Variables de entorno (ejemplo)

```bash
C2_MOTION_ENSEMBLE_RULE=mean
C2_MOTION_ENSEMBLE_THRESHOLD=0.50
# C2_MOTION_MODEL_A=.../modelo_49.pt   # o nombre Triton
# C2_MOTION_MODEL_B=.../modelo_53.pt
```

Desactivar / sobrescribir legacy:

- `C2_MOTION_ENSEMBLE_RULE=and`
- `C2_MOTION_ENSEMBLE_THRESHOLD=0.80`
- checkpoints **41** / **48**

---

## 8. Verificación (paridad C2 ↔ IAModel)

```bash
cd experiments/training/campaign
python score_pose_ensemble.py \
  --poses-full <chunk>/poses_full.npy \
  --valid-mask <chunk>/valid_mask.npy \
  --model-a <ruta>/modelo_49.pt \
  --model-b <ruta>/modelo_53.pt \
  --rule mean --threshold 0.50 --json
```

Comparar `p_robo` de cada modelo y `robbery_score` con el bloque equivalente en `p1.json` de C2 en el **mismo** chunk.

### Checklist de despliegue

- [ ] Dos `.pt` de celda **`bin_full`**, run `campaign_20261103_230735`
- [ ] `build_model`: **`tcn_gru`** (49) y **`conformer_lite`** (53)
- [ ] `num_classes=2`, probabilidad clase **índice 1**
- [ ] `poses_full` + política `valid_mask` alineada con train
- [ ] `seq_len=96` desde checkpoint
- [ ] Decisión: **MEAN ≥ 0.50**
- [ ] `robbery_score` en JSON = media, no `min`
- [ ] Smoke en ≥5 chunks prod vs `score_pose_ensemble.py`

---

## 9. TensorRT / Triton (si aplica)

- Un artefacto por checkpoint; preprocess en cliente hasta tensor `[1, 96, features]`.
- Nombres sugeridos: `motion_tcn_gru_exp49_binfull_v1`, `motion_conformer_exp53_binfull_v1`.
- Tras export, repetir paridad con chunks reales.

---

## 10. Impacto operativo

Más candidatos stage-1 que con **AND @ 0,80**. C3 / HAR / SAM3 deben asumir mayor volumen de `motion_theft_candidate`. Métricas: separar **candidato stage-1** vs **veredicto final C3**.

---

## 11. Referencias cruzadas

| Fichero | Uso |
|---------|-----|
| `score_pose_ensemble.py` | Ensemble + preprocess referencia |
| `train_model_operations.py` | `build_model`, preprocess |
| `model_config.py` | Catálogo EXPERIMENTS |
| `INFERENCE_ENSEMBLE_HANDOFF.md` | Plantilla handoff (otro par/celda) |
| `…/reports/bin_full/val_ensemble_grid.csv` | Fila mean 49\|53 @ 0,50 |

---

## Entregable pedido al agente C2

1. Triton / rutas `.pt` actualizados con 49 + 53.
2. Env `mean` + `0.50`.
3. Ejemplo `p1.json` con `robbery_score` = media.
4. Evidencia de paridad con `score_pose_ensemble.py` en chunks reales.

---

*IAModel — campaña `campaign_20261103_230735`, celda `bin_full`, marzo 2026.*