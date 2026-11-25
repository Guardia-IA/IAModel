"""Indexar ventanas 96-frame desde clips PoseExample (data_result)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np


@dataclass
class WindowExample:
    pose_path: Path
    valid_mask_path: Optional[Path]
    label: int
    clip_name: str
    category_str: str
    track_id: int
    action_label: Optional[int]
    start: int
    end: int  # exclusive; may be < start + window_len → pad
    window_id: str
    clip_uid: str
    forced_ops: Optional[list] = None
    # B1/B2/B3 metadata (set after indexing)
    label_state: str = "neg"
    train_label: int = 0
    sample_weight: float = 1.0
    pad_frac: float = 0.0
    core_overlap_frames: int = 0
    simulate_padded_negative: bool = False
    effective_start: Optional[int] = None
    effective_end: Optional[int] = None

    @property
    def window_len_effective(self) -> int:
        return self.end - self.start


def _clip_uid(ex) -> str:
    return f"{ex.category_str}::{ex.clip_name}::u{ex.track_id}"


def iter_window_slices(
    t: int,
    *,
    window_len: int,
    stride: int,
    include_short: bool = True,
    min_tail_for_final_window: int = 24,
) -> Iterator[tuple[int, int]]:
    if t <= 0:
        return
    if t < window_len:
        if include_short:
            yield 0, t
        return
    last_start = t - window_len
    starts: List[int] = []
    start = 0
    while start <= last_start:
        starts.append(start)
        start += max(1, stride)
    if starts:
        uncovered = t - (starts[-1] + window_len)
        if uncovered >= min_tail_for_final_window:
            if last_start not in starts:
                starts.append(last_start)
    for s in starts:
        yield s, s + window_len


def build_windows_from_examples(
    examples: list,
    *,
    window_len: int = 96,
    stride: int = 48,
    pose_source: str = "full",
    min_tail_for_final_window: int = 24,
) -> List[WindowExample]:
    out: List[WindowExample] = []
    for ex in examples:
        try:
            poses = np.load(ex.pose_path)
        except Exception:
            continue
        t = int(poses.shape[0])
        uid = _clip_uid(ex)
        for start, end in iter_window_slices(
            t,
            window_len=window_len,
            stride=stride,
            min_tail_for_final_window=min_tail_for_final_window,
        ):
            wid = f"{uid}::w{start:04d}"
            out.append(
                WindowExample(
                    pose_path=ex.pose_path,
                    valid_mask_path=getattr(ex, "valid_mask_path", None),
                    label=int(ex.label),
                    clip_name=str(ex.clip_name),
                    category_str=str(ex.category_str),
                    track_id=int(ex.track_id),
                    action_label=getattr(ex, "action_label", None),
                    start=int(start),
                    end=int(end),
                    window_id=wid,
                    clip_uid=uid,
                )
            )
    return out


def summarize_windows(windows: List[WindowExample]) -> dict:
    by_label: dict[int, int] = {}
    by_clip: set[str] = set()
    for w in windows:
        by_label[w.label] = by_label.get(w.label, 0) + 1
        by_clip.add(w.clip_uid)
    return {
        "n_windows": len(windows),
        "n_clips": len(by_clip),
        "by_label": dict(sorted(by_label.items())),
    }