"""Métricas binarias ventana / clip."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    n_neg = max(int((y_true == 0).sum()), 1)
    n_pos = max(int((y_true == 1).sum()), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision_pct": 100.0 * prec,
        "recall_pct": 100.0 * rec,
        "f1_pct": 100.0 * f1,
        "fp_rate_pct": 100.0 * fp / n_neg,
        "support_pos": n_pos,
        "support_neg": n_neg,
    }


def clip_metrics_from_window_probs(
    uids: list[str],
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Agrega max prob por uid (misma regla que P0.B)."""
    by_uid: Dict[str, list[float]] = {}
    by_y: Dict[str, int] = {}
    for u, yt, p in zip(uids, y_true, probs):
        by_uid.setdefault(u, []).append(float(p))
        by_y[u] = int(yt)
    clip_true: list[int] = []
    clip_pred: list[int] = []
    for u, plist in by_uid.items():
        clip_true.append(by_y[u])
        clip_pred.append(1 if max(plist) >= threshold else 0)
    return binary_metrics(np.array(clip_true), np.array(clip_pred))
