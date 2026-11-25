"""B1 — core-aware window labels and sample weights."""
from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core_detect import detect_core
from window_index import WindowExample


def _parse_cat(category_str: str) -> Optional[int]:
    try:
        return int(str(category_str).split("_")[0].split("/")[-1])
    except (ValueError, IndexError):
        for part in str(category_str).replace("\\", "/").split("/"):
            if part.isdigit():
                return int(part)
    return None


def load_core_index(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return dict(data.get("clips") or data)


def save_core_index(path: Path, clips: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "clips": clips}, f, indent=2)


def build_core_index_for_examples(examples: list, out_path: Path, robbery_cats: List[int]) -> Dict[str, Any]:
    clips: Dict[str, Any] = {}
    rob = {int(x) for x in robbery_cats}
    for ex in examples:
        cat = _parse_cat(ex.category_str)
        if cat not in rob:
            continue
        uid = f"{ex.category_str}::{ex.clip_name}::u{ex.track_id}"
        if uid in clips:
            continue
        core = detect_core(__import__("numpy").load(ex.pose_path))
        clips[uid] = core
    save_core_index(out_path, clips)
    return clips


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def apply_core_aware_labels(
    windows: List[WindowExample],
    core_index: Dict[str, Any],
    *,
    robbery_cats: List[int],
    cfg: Dict[str, Any],
) -> List[WindowExample]:
    pl = cfg.get("positive_labeling") or {}
    full_ov_cap = int(pl.get("full_ov_frames_cap", 36))
    full_ov_frac = float(pl.get("full_ov_frac", 0.6))
    partial_min_frac = float(pl.get("partial_min_frac", 0.2))
    partial_weight = float(pl.get("partial_weight", 0.3))
    rob = {int(x) for x in robbery_cats}
    lw = cfg.get("loss", {}).get("weights", {})
    w_neg = float(lw.get("neg", 1.0))
    w_pos_core = float(lw.get("pos_core", 2.0))
    pad_thr = float((cfg.get("padding") or {}).get("pad_weight_threshold", 0.25))
    pad_w = float((cfg.get("padding") or {}).get("pad_weight", 0.5))

    out: List[WindowExample] = []
    for w in windows:
        cat = _parse_cat(w.category_str)
        content_len = max(0, w.end - w.start)
        pad_frac = max(0.0, 1.0 - content_len / 96.0) if content_len else 1.0
        w = replace(w, pad_frac=pad_frac)

        if cat not in rob or int(w.label) != 1:
            w.label_state = "neg"
            w.train_label = 0
            w.sample_weight = w_neg
            if pad_frac > pad_thr:
                w.sample_weight *= pad_w
            out.append(w)
            continue

        core = core_index.get(w.clip_uid) or {}
        cs = int(core.get("core_start", 0))
        ce = int(core.get("core_end", w.end))
        core_len = max(1, ce - cs)
        ov = _overlap(w.start, w.end, cs, ce)
        need = min(int(full_ov_frac * core_len), full_ov_cap)

        if ov >= need:
            w.label_state = "pos_core"
            w.train_label = 1
            w.sample_weight = w_pos_core
            w.core_overlap_frames = ov
        elif ov >= int(partial_min_frac * core_len):
            w.label_state = "pos_partial"
            w.train_label = 1
            w.sample_weight = w_pos_core * partial_weight
            w.core_overlap_frames = ov
        else:
            w.label_state = "ignore"
            w.train_label = -1
            w.sample_weight = 0.0
            w.core_overlap_frames = ov

        if pad_frac > pad_thr and w.label_state != "ignore":
            w.sample_weight *= pad_w
        out.append(w)
    return out


def apply_padded_negative_simulation(
    windows: List[WindowExample],
    *,
    prob: float = 0.15,
    len_range: tuple[int, int] = (48, 95),
    seed: int = 42,
    window_len: int = 96,
) -> List[WindowExample]:
    """B2 — randomly truncate negatives and pad like short positives."""
    rng = random.Random(seed)
    lo, hi = len_range
    out: List[WindowExample] = []
    for w in windows:
        if w.label_state != "neg" or int(w.train_label) != 0:
            out.append(w)
            continue
        if rng.random() >= prob:
            out.append(w)
            continue
        content = w.end - w.start
        if content < lo:
            out.append(w)
            continue
        trunc = rng.randint(lo, min(hi, content))
        out.append(
            replace(
                w,
                simulate_padded_negative=True,
                effective_start=w.start,
                effective_end=w.start + trunc,
                pad_frac=max(0.0, 1.0 - trunc / window_len),
            )
        )
    return out