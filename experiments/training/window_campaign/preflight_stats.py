"""Estadísticas compartidas preflight ↔ entrenamiento."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from train_model_operations import PoseExample, split_examples_stratified_by_uid, SEED  # type: ignore

from core_labels import apply_core_aware_labels, apply_padded_negative_simulation, load_core_index
from window_index import WindowExample, build_windows_from_examples, summarize_windows


def _clip_window_counts(windows: List[WindowExample]) -> Tuple[int, int]:
    by_clip: Dict[str, int] = {}
    for w in windows:
        by_clip[w.clip_uid] = by_clip.get(w.clip_uid, 0) + 1
    multi = sum(1 for c in by_clip.values() if c > 1)
    return len(by_clip), multi


def estimate_train_rows_per_epoch(
    train_windows: List[WindowExample],
    *,
    extra_manifest_views: int,
    batch_size: int,
) -> int:
    trainable = [w for w in train_windows if w.label_state != "ignore" and w.sample_weight > 0]
    # identity + up to N manifest rows (upper bound if every clip had manifest)
    factor = 1 + max(0, int(extra_manifest_views))
    expanded = len(trainable) * factor
    return max(batch_size * 50, min(expanded, max(batch_size * 100, len(trainable))))


def build_preflight_stats(
    examples: List[PoseExample],
    cfg: Dict[str, Any],
    *,
    core_index: Dict[str, Any],
    data_root_str: str,
) -> Dict[str, Any]:
    wl = int(cfg.get("window_len", 96))
    stride = int(cfg.get("stride", 48))
    min_tail = int(cfg.get("min_tail_for_final_window", 24))
    split = cfg.get("split") or {}
    syn = cfg.get("synthetic_variants") or {}
    extra = int(syn.get("extra_manifest_views_per_clip", 1))
    pad_cfg = cfg.get("padding") or {}
    pl = cfg.get("positive_labeling") or {}
    batch_size = int((cfg.get("train_defaults") or {}).get("batch_size", 32))
    positive_classes = cfg.get("robbery_classes", [6, 14])

    train_clips, val_clips, test_clips = split_examples_stratified_by_uid(
        examples,
        seed=SEED,
        train_ratio=float(split.get("train_ratio", 0.8)),
        val_ratio=float(split.get("val_ratio", 0.1)),
    )

    def wins(clips: List[PoseExample]) -> List[WindowExample]:
        ws = build_windows_from_examples(
            clips,
            window_len=wl,
            stride=stride,
            min_tail_for_final_window=min_tail,
        )
        if pl.get("mode") == "core_aware" and core_index:
            ws = apply_core_aware_labels(ws, core_index, robbery_cats=positive_classes, cfg=cfg)
        return ws

    train_w = wins(train_clips)
    train_w = apply_padded_negative_simulation(
        train_w,
        prob=float(pad_cfg.get("padded_negative_prob", 0.15)),
        len_range=tuple(pad_cfg.get("padded_len_range", [48, 95])),
        window_len=wl,
    )
    val_w = wins(val_clips)
    test_w = wins(test_clips)

    all_w = build_windows_from_examples(
        examples,
        window_len=wl,
        stride=stride,
        min_tail_for_final_window=min_tail,
    )
    clips_total, clips_multi = _clip_window_counts(all_w)

    trainable = [w for w in train_w if w.label_state != "ignore" and w.sample_weight > 0]
    train_rows_syn = len(trainable) * (1 + extra)
    rows_per_epoch = estimate_train_rows_per_epoch(train_w, extra_manifest_views=extra, batch_size=batch_size)

    def _states(ws: List[WindowExample]) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for w in ws:
            c[w.label_state] = c.get(w.label_state, 0) + 1
        return c

    return {
        "data_root": data_root_str,
        "task": "binary",
        "clips": {
            "total": len(examples),
            "train": len(train_clips),
            "val": len(val_clips),
            "test": len(test_clips),
            "with_multiple_chunks": clips_multi,
            "with_single_chunk_only": clips_total - clips_multi,
        },
        "chunks_base": {
            "total_all_splits": len(all_w),
            "train": len(train_w),
            "val": len(val_w),
            "test": len(test_w),
            "by_label": summarize_windows(all_w)["by_label"],
        },
        "chunks_segmented_train": {
            "trainable_after_core": len(trainable),
            "ignored_stand_only": _states(train_w).get("ignore", 0),
            "pos_core": _states(train_w).get("pos_core", 0),
            "pos_partial": _states(train_w).get("pos_partial", 0),
            "neg": _states(train_w).get("neg", 0),
            "expanded_upper_bound_syn": int(train_rows_syn),
            "sampler_rows_per_epoch_est": rows_per_epoch,
        },
        "window": {"len": wl, "stride": stride, "min_tail": min_tail},
        "val_rows_per_epoch": len([w for w in val_w if w.label_state != "ignore"]),
        "train_rows_per_epoch": rows_per_epoch,
    }