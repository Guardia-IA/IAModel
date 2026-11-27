"""Campaign2 labels: pos >= 80% core overlap; other robbery windows -> ignore (never neg)."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

from campaign2_paths import CAMPAIGN2_ROOT

# Reuse window_campaign types
import sys

_WC = CAMPAIGN2_ROOT.parent / "window_campaign"
if str(_WC) not in sys.path:
    sys.path.insert(0, str(_WC))

from core_labels import (  # type: ignore
    _overlap,
    _parse_cat,
    build_core_index_for_examples,
    load_core_index,
    save_core_index,
)
from window_index import WindowExample  # type: ignore


def apply_campaign2_labels(
    windows: List[WindowExample],
    core_index: Dict[str, Any],
    *,
    robbery_cats: List[int],
    cfg: Dict[str, Any],
) -> List[WindowExample]:
    pl = cfg.get("positive_labeling") or {}
    frac_pos = float(pl.get("core_overlap_frac_pos", 0.8))
    rob = {int(x) for x in robbery_cats}
    lw = cfg.get("loss", {}).get("weights", {})
    w_neg = float(lw.get("neg", 1.0))
    w_pos_core = float(lw.get("pos_core", 2.0))
    pad_thr = float((cfg.get("padding") or {}).get("pad_weight_threshold", 0.25))
    pad_w = float((cfg.get("padding") or {}).get("pad_weight", 0.5))
    wl = max(1, int(cfg.get("window_len", 96)))

    out: List[WindowExample] = []
    for w in windows:
        cat = _parse_cat(w.category_str)
        content_len = max(0, w.end - w.start)
        pad_frac = max(0.0, 1.0 - content_len / float(wl)) if content_len else 1.0
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
        need = int(frac_pos * core_len)

        if ov >= need:
            w.label_state = "pos_core"
            w.train_label = 1
            w.sample_weight = w_pos_core
            w.core_overlap_frames = ov
            if pad_frac > pad_thr:
                w.sample_weight *= pad_w
        else:
            # Standing before/after grab, partial overlap: never train as negative.
            w.label_state = "ignore"
            w.train_label = -1
            w.sample_weight = 0.0
            w.core_overlap_frames = ov

        out.append(w)
    return out


def ensure_core_index_file(cfg: Dict[str, Any], examples: list) -> Dict[str, Any]:
    from campaign2_paths import ensure_core_index

    path = ensure_core_index(cfg)
    robbery = cfg.get("robbery_classes", [6, 14])
    if not path.is_file():
        build_core_index_for_examples(examples, path, robbery)
    return load_core_index(path)