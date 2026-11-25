# Handoff C2 — Despliegue stage-1 robo (MEAN 48 + 49 @ 0,50)

Documento para el **agente de inferencia C2** (`c2_inference_service` / Triton motion).

**Par desplegado:** `modelo_48.pt` + `modelo_49.pt` (misma familia **`tcn_gru`**, **dos grafos distintos** — no intercambiar pesos).

---

## Resumen en una frase

```text
robbery_score = ( P_robo(modelo_48) + P_robo(modelo_49) ) / 2
alarma_robo     = robbery_score >= 0.50
```

Regla: **`mean`** (no `and`). Umbral: **`0.50`**.

**Referencia val (`bin_full`):** F1 ~**86,3 %**, recall ~**82,4 %** (por debajo del par 49+53 ~88 % F1, aceptable si ya copiaste 48|49).

---

## 1. Configuración de producción

| Campo | Valor |
|--------|--------|
| **Campaña** | `campaign_20261103_230735` |
| **Celda** | **`bin_full`** |
| **Tarea** | Binario: clase **1** = robo (`softmax(logits)[1]`) |
| **Entrada** | **`poses_full.npy`** + **`valid_mask.npy`** si existe |
| **Slot 1** (`p1` / `model.pt`) | **`modelo_48.pt`** |
| **Slot 2** (`p2` / `model2.pt`) | **`modelo_49.pt`** |
| **Regla** | **`mean`** |
| **Umbral** | **`0.50`** |

No usar AND @ 0,80 ni `robbery_score = min(p1, p2)` salvo cambio explícito de producto.

---

## 2. Arquitecturas (ambas `tcn_gru`, hiperparámetros distintos)

Cargar **`ckpt["config"]`** por fichero. Constructor: `PoseTCNGRUClassifier` en `train_model_operations.py`.

### Exp 48 — `modelo_48.pt`

| Parámetro | Valor |
|-----------|--------|
| `arch` | `tcn_gru` |
| `tcn_hidden_dim` | 128 |
| `tcn_layers` | **2** |
| `gru_hidden_dim` | 128 |
| `gru_layers` | 1 |
| `dropout` | 0.2 |
| **`seq_len`** | **96** |

### Exp 49 — `modelo_49.pt`

| Parámetro | Valor |
|-----------|--------|
| `arch` | `tcn_gru` |
| `tcn_hidden_dim` | 128 |
| `tcn_layers` | **3** |
| `gru_hidden_dim` | 128 |
| `gru_layers` | 1 |
| `dropout` | 0.25 |
| **`seq_len`** | **96** |

**Crítico:** mismo nombre de arquitectura en Triton/manifest, pero **dos modelos separados** (48 ≠ 49 en profundidad TCN y dropout). Cada slot debe hacer `build_model` con su propio `config` del `.pt`.

Catálogo: `model_config.py` → `EXPERIMENTS[47]` (exp 48), `EXPERIMENTS[48]` (exp 49).

---

## 3. Rutas de checkpoints

```text
…/campaign_20261103_230735/models/bin_full/modelo_48.pt
…/campaign_20261103_230735/models/bin_full/modelo_49.pt
```

Referencia Angel:

```text
/home/angel/IAModel/experiments/training/campaign/artifacts/runs/campaign_20261103_230735/models/bin_full/modelo_48.pt
/home/angel/IAModel/experiments/training/campaign/artifacts/runs/campaign_20261103_230735/models/bin_full/modelo_49.pt
```

---

## 4. Carga e inferencia

```python
ckpt = torch.load(path, map_location=device, weights_only=False)
cfg = dict(ckpt["config"])
model = build_model(cfg["arch"], int(ckpt["input_dim"]), int(ckpt.get("num_classes", 2)), cfg)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

probs = F.softmax(model(x), dim=1)
p_robo = float(probs[0, 1].item())
```

Preprocess (igual en ambos slots, mismo buffer de poses):

1. `poses_full` → filtrar con `valid_mask` si existe  
2. NaN → 0 → `normalize_sequence` → `add_velocity` → `temporal_resize(..., seq_len=96)`  
3. Tensor `[1, T, features]`

---

## 5. Fusión y `p1.json`

```python
C2_MOTION_ENSEMBLE_RULE = "mean"
C2_MOTION_ENSEMBLE_THRESHOLD = 0.50

p48 = forward_p_robo(model_48, poses, device)
p49 = forward_p_robo(model_49, poses, device)
robbery_score = (p48 + p49) / 2.0
robbery_alarm = robbery_score >= 0.50
```

- `p1` → modelo **48** (`arch=tcn_gru`, `tcn_layers=2`)  
- `p2` → modelo **49** (`arch=tcn_gru`, `tcn_layers=3`)  
- `robbery_score` = **media**, no `min`

---

## 6. Variables de entorno (ejemplo)

```bash
C2_MOTION_ENSEMBLE_RULE=mean
C2_MOTION_ENSEMBLE_THRESHOLD=0.50
```

Sustituye legacy 41+48 AND @ 0,80.

---

## 7. Verificación

```bash
cd experiments/training/campaign
python score_pose_ensemble.py \
  --poses-full <chunk>/poses_full.npy \
  --valid-mask <chunk>/valid_mask.npy \
  --model-a <ruta>/modelo_48.pt \
  --model-b <ruta>/modelo_49.pt \
  --rule mean --threshold 0.50 --json
```

Comparar `p48`/`p49` y `robbery_score` con C2 en el mismo chunk.

### Checklist

- [ ] Slot A = **48**, slot B = **49** (orden coherente en `p1.json`)
- [ ] Dos instancias `tcn_gru` con **configs distintas** (2 vs 3 capas TCN)
- [ ] MEAN @ **0.50**
- [ ] Celda **`bin_full`**, misma campaña

---

## 8. Nota vs par 49 + 53

Si más adelante queréis el máximo F1 en val (~**88,2 %**), el estudio IAModel recomienda **49 + conformer_lite (53)** en MEAN @ 0,50. Con **48 + 49** homogéneos ganáis despliegue más simple (misma familia, dos engines similares) a cambio de ~**2 pp** menos F1 en val.

---

*IAModel — `campaign_20261103_230735`, celda `bin_full`.*