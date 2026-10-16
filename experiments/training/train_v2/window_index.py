"""Índice de ventanas sobre manifest Campaign2 (12 Hz)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from train_v2.legacy_preprocess import window_starts


@dataclass(frozen=True)
class WindowRef:
    entry_index: int
    start_frame: int
    uid: str
    group_id: str
    label_theft: int


def build_window_index(
    entries: List[Dict[str, Any]],
    *,
    split: str,
    window_frames: int,
    stride_frames: int,
    robbery_class: int = 6,
    max_windows_per_clip: int = 0,
) -> List[WindowRef]:
    out: List[WindowRef] = []
    for idx, ent in enumerate(entries):
        if str(ent.get("split")) != split:
            continue
        t = int(ent.get("num_frames") or 0)
        if t <= 0:
            continue
        starts = window_starts(t, window_frames, stride_frames)
        if max_windows_per_clip > 0 and len(starts) > max_windows_per_clip:
            step = max(1, len(starts) // max_windows_per_clip)
            starts = starts[::step][:max_windows_per_clip]
        is_rob = bool(ent.get("is_robbery")) or int(ent.get("label", -1)) == int(robbery_class)
        label = 1 if is_rob else 0
        uid = str(ent.get("uid") or idx)
        gid = str(ent.get("group_id") or uid)
        for s in starts:
            out.append(
                WindowRef(
                    entry_index=idx,
                    start_frame=int(s),
                    uid=uid,
                    group_id=gid,
                    label_theft=label,
                )
            )
    return out
