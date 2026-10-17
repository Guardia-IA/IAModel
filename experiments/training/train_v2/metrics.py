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


def aggregate_clip_scores(
    uids: list[str],
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    clip_y_by_uid: Dict[str, int] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Max prob por clip (uid). Etiqueta clip = is_robbery del manifest si se pasa."""
    by_uid: Dict[str, list[float]] = {}
    by_y: Dict[str, int] = {}
    for u, yt, p in zip(uids, y_true, probs):
        by_uid.setdefault(u, []).append(float(p))
        by_y[u] = max(by_y.get(u, 0), int(yt))
    clip_uids = sorted(by_uid.keys())
    clip_y = np.array(
        [
            int(clip_y_by_uid.get(u, by_y[u])) if clip_y_by_uid else int(by_y[u])
            for u in clip_uids
        ],
        dtype=np.int64,
    )
    clip_p = np.array([max(by_uid[u]) for u in clip_uids], dtype=np.float64)
    return clip_uids, clip_y, clip_p


def threshold_from_negative_order_statistic(
    clip_y: np.ndarray,
    clip_probs: np.ndarray,
    *,
    max_fp_clips: int,
) -> float:
    """
    Umbral en split T: ordena scores de sesiones/clips negativos (desc) y elige
    el estadístico de orden que permite como máximo max_fp_clips FP en negativos.
    """
    neg = clip_probs[clip_y == 0]
    if neg.size == 0:
        return 0.5
    k = int(max(0, max_fp_clips))
    if k <= 0:
        return float(np.nextafter(1.0, 0.0))
    neg_sorted = np.sort(neg)[::-1]
    if k >= neg_sorted.size:
        return 0.0
    return float(neg_sorted[k - 1])


def clip_metrics_at_fp_budget(
    uids: list[str],
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    max_fp_clips: int,
    threshold: float | None = None,
    clip_y_by_uid: Dict[str, int] | None = None,
) -> Dict[str, Any]:
    """Recall clip-level con umbral fijado por presupuesto FP en negativos."""
    _, clip_y, clip_p = aggregate_clip_scores(
        uids, y_true, probs, clip_y_by_uid=clip_y_by_uid
    )
    if threshold is None:
        threshold = threshold_from_negative_order_statistic(clip_y, clip_p, max_fp_clips=max_fp_clips)
    pred = (clip_p >= threshold).astype(np.int64)
    m = binary_metrics(clip_y, pred)
    fp = int(m["fp"])
    m["threshold"] = float(threshold)
    m["max_fp_clips"] = int(max_fp_clips)
    m["selection_score"] = float(m["recall_pct"]) - max(0, fp - int(max_fp_clips)) * 5.0
    return m


def clip_metrics_from_window_probs(
    uids: list[str],
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
    clip_y_by_uid: Dict[str, int] | None = None,
) -> Dict[str, Any]:
    """Agrega max prob por uid (misma regla que P0.B)."""
    clip_uids, clip_y, clip_p = aggregate_clip_scores(
        uids, y_true, probs, clip_y_by_uid=clip_y_by_uid
    )
    clip_pred = (clip_p >= threshold).astype(np.int64)
    return binary_metrics(clip_y, clip_pred)
