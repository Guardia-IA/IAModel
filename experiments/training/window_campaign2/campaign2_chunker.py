"""Production-like 96-frame chunks (stride 48) and cross-segment overlap windows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np

from window_index import iter_window_slices  # type: ignore


@dataclass
class ChunkWindow:
    start: int
    end: int
    is_overlap: bool = False
    source: str = "in_file"


def iter_production_windows(
    n_frames: int,
    *,
    window_len: int = 96,
    stride: int = 48,
    min_tail: int = 24,
) -> Iterator[ChunkWindow]:
    for s, e in iter_window_slices(
        n_frames,
        window_len=window_len,
        stride=stride,
        include_short=True,
        min_tail_for_final_window=min_tail,
    ):
        yield ChunkWindow(start=s, end=e, is_overlap=False, source="in_file")


def concat_overlap_window(
    poses_a: np.ndarray,
    poses_b: np.ndarray,
    *,
    window_len: int = 96,
    half: int = 48,
) -> np.ndarray:
    """tail(a, half) + head(b, half) == window_len rows."""
    tail = poses_a[-half:] if len(poses_a) >= half else poses_a
    head = poses_b[:half] if len(poses_b) >= half else poses_b
    if len(tail) < half:
        pad = np.repeat(tail[-1:], half - len(tail), axis=0) if len(tail) else np.zeros((half - len(tail), *poses_a.shape[1:]))
        tail = np.concatenate([tail, pad], axis=0)
    if len(head) < half:
        pad = np.repeat(head[-1:], half - len(head), axis=0) if len(head) else np.zeros((half - len(head), *poses_b.shape[1:]))
        head = np.concatenate([head, pad], axis=0)
    return np.concatenate([tail, head], axis=0).astype(np.float32)