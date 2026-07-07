"""Calibración por temperatura en val (post-hoc)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from train_model_operations import build_model, build_pose_dataset_for_eval
except ImportError:
    from ..train_model_operations import build_model, build_pose_dataset_for_eval  # type: ignore


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    best_t = 1.0
    best_nll = float("inf")
    for t in np.linspace(0.5, 3.0, 51):
        nll = F.cross_entropy(logits / float(t), labels, reduction="mean").item()
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t


def collect_val_logits(
    model_path: Path,
    examples: List[Any],
    *,
    label_to_idx: Dict[int, int],
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    seq_len = int(ckpt.get("seq_len", 64))
    cfg = ckpt.get("config", {})
    arch = cfg.get("arch", "tcn")
    input_dim = int(ckpt["input_dim"])
    num_classes = int(ckpt.get("num_classes", len(label_to_idx)))

    ds = build_pose_dataset_for_eval(
        examples, label_to_idx, seq_len, dataset_split="val", checkpoint=ckpt,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model = build_model(arch, input_dim, num_classes, cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logits_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            logits_all.append(model(x.to(device)).cpu())
            labels_all.append(y.cpu())
    logits = torch.cat(logits_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    return logits, labels, []


def calibrate_model_on_val(
    model_path: Path,
    examples: List[Any],
    *,
    label_to_idx: Dict[int, int],
    task: str,
    robbery_class: int = 6,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Ajusta temperatura en val y devuelve métricas calibradas + umbral óptimo FP."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logits, labels, _ = collect_val_logits(
        model_path, examples, label_to_idx=label_to_idx, device=device,
    )
    temperature = fit_temperature(logits, labels)

    if task == "binary" or logits.shape[1] == 2:
        probs = torch.softmax(logits / temperature, dim=1)[:, 1].numpy()
        y_true = (labels.numpy() == 1).astype(np.int64)
    else:
        idx_pos = label_to_idx.get(int(robbery_class), 1)
        probs = torch.softmax(logits / temperature, dim=1)[:, int(idx_pos)].numpy()
        y_true = (labels.numpy() == int(idx_pos)).astype(np.int64)

    best_thr = 0.5
    best_f1 = -1.0
    best_row: Dict[str, Any] = {}
    for thr in np.arange(0.5, 0.99, 0.02):
        y_pred = (probs >= thr).astype(np.int64)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        n_neg = max(fp + tn, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_row = {
                "threshold": best_thr,
                "f1_pct": 100.0 * f1,
                "recall_pct": 100.0 * rec,
                "fp_rate_pct": 100.0 * fp / n_neg,
                "fp": fp,
                "fn": fn,
            }

    return {
        "temperature": float(temperature),
        "calibrated_threshold": best_thr,
        **best_row,
    }
