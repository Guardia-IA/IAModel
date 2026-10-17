"""Estadísticas de clips, splits y ventanas train_v2 para preflight."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from train_v2.window_index import build_window_index


SPLIT_ORDER = ("train", "val_dev", "threshold_calib", "frozen_test")

SPLIT_LABELS = {
    "train": "Entrenamiento (train)",
    "val_dev": "Validación desarrollo (val_dev)",
    "threshold_calib": "Calibración umbral T",
    "frozen_test": "Test congelado (frozen_test)",
}


def summarize_training_dataset(
    entries: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    *,
    debug_mode: bool = False,
) -> Dict[str, Any]:
    temporal = cfg.get("temporal") or {}
    p1 = cfg.get("p1_train") or {}
    split_cfg = cfg.get("split") or {}
    robbery_class = int(cfg.get("robbery_class", 6))

    target_hz = float(temporal.get("target_hz", 12.0))
    window_seconds = float(temporal.get("window_seconds", 8.0))
    stride_seconds = float(temporal.get("stride_seconds", 1.0))
    window_frames = max(1, int(round(window_seconds * target_hz)))
    stride_frames = max(1, int(round(stride_seconds * target_hz)))
    max_w = int(p1.get("max_windows_per_clip", 32))
    if debug_mode:
        dbg_max = (cfg.get("debug") or {}).get("max_windows_per_clip")
        if dbg_max is not None and int(dbg_max) > 0:
            max_w = int(dbg_max)

    policy = str(p1.get("window_label_policy", "tail_positive"))
    tail_sec = float(p1.get("robbery_positive_tail_seconds", 3.0))
    augment_enabled = bool(p1.get("augment_enabled", True))

    category_global: Dict[str, int] = {
        str(k): v for k, v in sorted(Counter(int(e.get("label", -1)) for e in entries).items()) if k >= 0
    }

    split_clips: Dict[str, Dict[str, int]] = {}
    category_by_split: Dict[str, Dict[str, int]] = {}
    windows_by_split: Dict[str, int] = {}
    windows_pos_by_split: Dict[str, int] = {}
    windows_neg_by_split: Dict[str, int] = {}

    for sp in SPLIT_ORDER:
        split_entries = [e for e in entries if str(e.get("split")) == sp]
        if not split_entries and sp not in {str(e.get("split")) for e in entries}:
            continue
        rob = sum(1 for e in split_entries if e.get("is_robbery") or int(e.get("label", 0)) == robbery_class)
        split_clips[sp] = {"clips": len(split_entries), "robbery": rob}
        cat = Counter(int(e.get("label", -1)) for e in split_entries)
        category_by_split[sp] = {str(k): v for k, v in sorted(cat.items()) if k >= 0}

        refs = build_window_index(
            entries,
            split=sp,
            window_frames=window_frames,
            stride_frames=stride_frames,
            robbery_class=robbery_class,
            max_windows_per_clip=max_w,
            target_hz=target_hz,
            window_label_policy=policy,
            robbery_positive_tail_seconds=tail_sec,
        )
        n_pos = sum(1 for r in refs if int(r.label_theft) >= 1)
        windows_by_split[sp] = len(refs)
        windows_pos_by_split[sp] = n_pos
        windows_neg_by_split[sp] = len(refs) - n_pos

    ratios = dict(split_cfg.get("ratios") or {})
    return {
        "num_clips_total": len(entries),
        "category_counts": category_global,
        "category_counts_by_split": category_by_split,
        "split_clips": split_clips,
        "split_ratios_config": ratios,
        "windows": {
            "window_seconds": window_seconds,
            "stride_seconds": stride_seconds,
            "target_hz": target_hz,
            "max_windows_per_clip": max_w,
            "window_label_policy": policy,
            "by_split": windows_by_split,
            "positive_labels_by_split": windows_pos_by_split,
            "negative_labels_by_split": windows_neg_by_split,
        },
        "augment": {
            "enabled": augment_enabled,
            "mode": "on_the_fly_train_only",
            "note_es": (
                "Campaign2 no duplica clips en disco (no mass-augment legacy). "
                "Las ventanas 8 s indexadas son muestras reales; con augment_enabled "
                "cada época aplica flip/jitter/dropout aleatorio sobre train (variantes sintéticas ilimitadas)."
            ),
        },
    }
