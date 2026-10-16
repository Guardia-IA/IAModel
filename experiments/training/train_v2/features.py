"""Features por ventana (MVP P1: norm legacy + velocidad + validez por joint)."""
from __future__ import annotations

import numpy as np

from train_v2.legacy_preprocess import add_velocity, normalize_sequence
from train_v2.normalize import normalize_torso_isotropic

PREPROCESS_LEGACY = "campaign2_windows_legacy_norm_v1"
PREPROCESS_TORSO = "campaign2_windows_torso_v1"


def window_to_features(
    poses: np.ndarray,
    valid: np.ndarray,
    *,
    seq_len: int,
    min_valid_joint_ratio: float = 0.25,
    normalize_mode: str = PREPROCESS_TORSO,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    poses: [T, J, 2], valid: [T, J] bool/0-1.
    Returns x [seq_len, F], frame_mask [seq_len], ok.
    """
    poses = np.asarray(poses, dtype=np.float32)
    valid = np.asarray(valid, dtype=np.float32)
    t, j, _ = poses.shape
    if t <= 0:
        return np.zeros((seq_len, j * 5), dtype=np.float32), np.zeros(seq_len, dtype=np.float32), False

    if np.any(~np.isfinite(poses)):
        poses = np.nan_to_num(poses, nan=0.0)

    vr = float(valid.mean()) if valid.size else 0.0
    if vr < min_valid_joint_ratio:
        return np.zeros((seq_len, j * 5), dtype=np.float32), np.zeros(seq_len, dtype=np.float32), False

    if normalize_mode in (PREPROCESS_LEGACY, "legacy_mean", "legacy"):
        poses = normalize_sequence(poses)
    else:
        poses = normalize_torso_isotropic(poses, valid)
    poses = add_velocity(poses)
    valid_t = valid.mean(axis=1).astype(np.float32)
    if t != seq_len:
        idx = np.linspace(0, t - 1, seq_len).round().astype(int)
        poses = poses[idx]
        valid = valid[idx]
        valid_t = valid_t[idx]

    feat = np.concatenate([poses, valid[..., None]], axis=-1)
    feat = feat.reshape(seq_len, j * 5).astype(np.float32)
    frame_mask = (valid_t > 0.5).astype(np.float32)
    return feat, frame_mask, True
