"""Window-level and event-level metrics for campaign2."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def binary_window_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    y = y.astype(bool)
    pred = pred.astype(bool)
    pos = y
    neg = ~pos
    tp = int((pred & pos).sum())
    fp = int((pred & neg).sum())
    fn = int((~pred & pos).sum())
    tn = int((~pred & neg).sum())
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    rec = 100.0 * tp / max(n_pos, 1)
    fpr = 100.0 * fp / max(n_neg, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "recall_pct": rec,
        "fp_rate_pct": fpr,
    }


def threshold_sweep(
    y: np.ndarray,
    probs: np.ndarray,
    thresholds: Sequence[float],
) -> List[Dict[str, float]]:
    rows = []
    for thr in thresholds:
        m = binary_window_metrics(y, probs >= thr)
        m["threshold"] = thr
        rows.append(m)
    return rows


def event_recall_or(
    *,
    clip_uids: Sequence[str],
    clip_is_positive: Sequence[bool],
    window_probs: Sequence[float],
    window_clip_uid: Sequence[str],
    threshold: float,
) -> Dict[str, float]:
    """One positive event = any window from that clip alarms."""
    by_clip: Dict[str, List[float]] = defaultdict(list)
    pos_clips = set()
    for uid, is_pos, p in zip(clip_uids, clip_is_positive, window_probs):
        by_clip[uid].append(float(p))
        if is_pos:
            pos_clips.add(uid)

    detected = 0
    for uid in pos_clips:
        if any(p >= threshold for p in by_clip.get(uid, [])):
            detected += 1
    n_pos = len(pos_clips)
    return {
        "n_events_pos": n_pos,
        "events_detected": detected,
        "event_recall_pct": 100.0 * detected / max(n_pos, 1),
        "threshold": threshold,
    }


def fp_buckets_in_robbery_files(
    windows_meta: List[Dict],
    probs: np.ndarray,
    threshold: float,
) -> Dict[str, Dict[str, float]]:
    """P0-1: FP on neg-folder vs in-robbery-file windows."""
    buckets: Dict[str, List[int]] = defaultdict(list)
    for i, meta in enumerate(windows_meta):
        if meta.get("train_label", 0) == 1:
            continue
        src = meta.get("neg_source", "neg_folder")
        if src == "in_robbery_file":
            dist = float(meta.get("core_distance_sec", 0.0))
            if dist < 2.0:
                key = "rob_file_near_core"
            elif dist < 5.0:
                key = "rob_file_mid"
            else:
                key = "rob_file_far"
        else:
            key = "neg_folder"
        buckets[key].append(i)

    out: Dict[str, Dict[str, float]] = {}
    for key, idxs in buckets.items():
        sub_y = np.zeros(len(idxs), dtype=np.int64)
        sub_p = (probs[idxs] >= threshold).astype(np.int64)
        m = binary_window_metrics(sub_y, sub_p)
        m["count_windows"] = len(idxs)
        out[key] = m
    return out


def recall_at_fpr_score(val_metrics: Dict, target_fpr_pct: float, min_recall_pct: float) -> float:
    """Checkpoint score: maximize recall with FP <= target; penalize if recall below floor."""
    block = (val_metrics.get("binary") or {}).get("softmax_argmax") or {}
    if not block:
        block = (val_metrics.get("binary") or {}).get("softmax_threshold") or {}
    rec = float(block.get("recall_robbery_pct", 0.0))
    fpr = float(block.get("false_positive_rate_pct", 100.0))
    if rec <= 0.0:
        return -1.0
    if rec < min_recall_pct:
        return -1.0
    if fpr > target_fpr_pct:
        return rec - (fpr - target_fpr_pct)
    return rec + 0.01 * (target_fpr_pct - fpr)