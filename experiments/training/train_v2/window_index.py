"""Índice de ventanas sobre manifest Campaign2 (12 Hz)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from train_v2.legacy_preprocess import window_starts
from train_v2.window_labels import POLICY_CLIP_ALL, window_theft_label


@dataclass(frozen=True)
class WindowRef:
    entry_index: int
    start_frame: int
    uid: str
    group_id: str
    label_theft: int
    clip_label: int = 0
    clip_is_robbery: int = 0


def build_window_index(
    entries: List[Dict[str, Any]],
    *,
    split: str,
    window_frames: int,
    stride_frames: int,
    robbery_class: int = 6,
    max_windows_per_clip: int = 0,
    target_hz: float = 12.0,
    window_label_policy: str = POLICY_CLIP_ALL,
    robbery_positive_tail_seconds: float = 3.0,
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
        clip_label = int(ent.get("label", 0))
        is_rob = bool(ent.get("is_robbery")) or clip_label == int(robbery_class)
        uid = str(ent.get("uid") or idx)
        gid = str(ent.get("group_id") or uid)
        for s in starts:
            label = window_theft_label(
                is_robbery_clip=is_rob,
                start_frame=int(s),
                window_frames=window_frames,
                num_frames=t,
                target_hz=float(target_hz),
                policy=window_label_policy,
                tail_seconds=robbery_positive_tail_seconds,
            )
            out.append(
                WindowRef(
                    entry_index=idx,
                    start_frame=int(s),
                    uid=uid,
                    group_id=gid,
                    label_theft=label,
                    clip_label=clip_label,
                    clip_is_robbery=1 if is_rob else 0,
                )
            )
    return out
