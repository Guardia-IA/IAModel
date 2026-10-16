"""Augment on-the-fly en ventanas (solo train)."""
from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np


def augment_pose_window(
    poses: np.ndarray,
    valid: np.ndarray,
    rng: random.Random,
    cfg: Dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Operaciones no nulas bajo normalización posterior: flip horizontal, jitter temporal,
    dropout de joints y gaps cortos de visibilidad.
    """
    cfg = cfg or {}
    p = float(cfg.get("prob", 0.5))
    if rng.random() > p:
        return poses, valid

    poses = np.asarray(poses, dtype=np.float32).copy()
    valid = np.asarray(valid, dtype=np.float32).copy()

    if cfg.get("flip_x", True) and rng.random() < float(cfg.get("p_flip", 0.5)):
        poses[..., 0] = 1.0 - poses[..., 0]

    speed_lo = float(cfg.get("speed_min", 0.9))
    speed_hi = float(cfg.get("speed_max", 1.1))
    if cfg.get("speed_jitter", True) and poses.shape[0] > 2:
        factor = rng.uniform(speed_lo, speed_hi)
        t = poses.shape[0]
        new_len = max(2, int(round(t * factor)))
        idx = np.linspace(0, t - 1, new_len).round().astype(int)
        poses = poses[idx]
        valid = valid[idx]
        if new_len < t:
            pad = t - new_len
            poses = np.concatenate([poses, np.repeat(poses[-1:], pad, axis=0)], axis=0)
            valid = np.concatenate([valid, np.repeat(valid[-1:], pad, axis=0)], axis=0)
        elif new_len > t:
            poses = poses[:t]
            valid = valid[:t]

    j, = valid.shape[1:]
    if cfg.get("joint_dropout", True):
        p_drop = float(cfg.get("p_joint_dropout", 0.08))
        for jj in range(j):
            if rng.random() < p_drop:
                valid[:, jj] = 0.0

    if cfg.get("gap_frames", True):
        p_gap = float(cfg.get("p_gap", 0.15))
        if rng.random() < p_gap and poses.shape[0] >= 4:
            gap_len = rng.randint(1, max(1, poses.shape[0] // 8))
            start = rng.randint(0, poses.shape[0] - gap_len)
            valid[start : start + gap_len] = 0.0

    return poses, valid
