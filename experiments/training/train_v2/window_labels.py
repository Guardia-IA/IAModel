"""Políticas de etiqueta por ventana (MIL-lite, experto §4)."""
from __future__ import annotations

POLICY_CLIP_ALL = "clip_all"
POLICY_TAIL_POSITIVE = "tail_positive"


def window_theft_label(
    *,
    is_robbery_clip: bool,
    start_frame: int,
    window_frames: int,
    num_frames: int,
    target_hz: float,
    policy: str = POLICY_TAIL_POSITIVE,
    tail_seconds: float = 3.0,
) -> int:
    """
    clip_all: todas las ventanas heredan etiqueta clip (MIL débil antiguo).
    tail_positive: solo ventanas que solapan la cola final del clip (gesto típico).
    """
    if not is_robbery_clip:
        return 0
    if policy == POLICY_CLIP_ALL or target_hz <= 0 or num_frames <= 0:
        return 1

    duration_sec = max((num_frames - 1) / float(target_hz), 0.0)
    tail_sec = max(float(tail_seconds), 0.0)
    if duration_sec <= tail_sec + 1e-6:
        return 1

    win_start_sec = start_frame / float(target_hz)
    win_end_sec = (start_frame + window_frames) / float(target_hz)
    tail_start_sec = duration_sec - tail_sec
    overlaps_tail = win_end_sec > tail_start_sec and win_start_sec <= duration_sec + 1e-6
    return 1 if overlaps_tail else 0
