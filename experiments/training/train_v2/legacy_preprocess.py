"""Preprocesado legacy (normalize → velocity → temporal_resize) para eval P0."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TRAINING_DIR = Path(__file__).resolve().parent.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from train_model_operations import add_velocity, normalize_sequence, temporal_resize  # noqa: E402


def poses_to_model_input(
    poses: np.ndarray,
    seq_len: int,
    *,
    pad_to_len: int | None = None,
) -> np.ndarray:
    """
    poses: [T, J, 2] float (NaN permitidos → 0).
    Devuelve [seq_len, J*4] float32 (misma lógica que PoseDataset en val).
    """
    poses = np.asarray(poses, dtype=np.float32)
    if np.any(~np.isfinite(poses)):
        poses = np.nan_to_num(poses, nan=0.0, posinf=0.0, neginf=0.0)
    if pad_to_len is not None and poses.shape[0] < pad_to_len:
        pad = np.repeat(poses[-1:], pad_to_len - poses.shape[0], axis=0)
        poses = np.concatenate([poses, pad], axis=0)
    elif pad_to_len is not None and poses.shape[0] > pad_to_len:
        poses = poses[:pad_to_len]

    poses = normalize_sequence(poses)
    poses = add_velocity(poses)
    poses = temporal_resize(poses, seq_len)
    t, j, d = poses.shape
    return poses.reshape(t, j * d).astype(np.float32)


def window_starts(num_frames: int, window_frames: int, stride_frames: int) -> list[int]:
    if num_frames <= 0 or window_frames <= 0:
        return []
    if num_frames < window_frames:
        return [0]
    last = num_frames - window_frames
    starts = list(range(0, last + 1, max(1, stride_frames)))
    if starts[-1] != last and last > starts[-1]:
        starts.append(last)
    return starts
