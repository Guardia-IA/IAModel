"""Corte / padding de ventanas + keep_grid_v2 validity (B5)."""
from __future__ import annotations

import numpy as np

MIN_VISIBLE_JOINTS = 5


def visible_joint_count(poses: np.ndarray) -> np.ndarray:
    """poses [T,J,2] -> count visible joints per frame."""
    fin = np.isfinite(poses).all(axis=-1)
    nonzero = np.abs(poses).sum(axis=-1) > 1e-7
    return (fin & nonzero).sum(axis=1).astype(np.int32)


def apply_keep_grid_v2(poses: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Mark low-visibility frames; zero invalid frames; velocity gating done in add_velocity_v2.
    poses [T,J,2] in place logic on copy.
    """
    out = np.nan_to_num(poses, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    vis = visible_joint_count(out)
    if valid_mask is not None:
        vm = valid_mask.astype(bool)
        if len(vm) == len(out):
            vis = np.minimum(vis, vm.astype(np.int32) * out.shape[1])
    frame_ok = vis >= MIN_VISIBLE_JOINTS
    out[~frame_ok] = 0.0
    return out


def normalize_visible_sequence(poses: np.ndarray) -> np.ndarray:
    """Centra y escala solo joints visibles. Los frames a cero no mueven la media."""
    out = np.nan_to_num(poses, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    vis = np.abs(out).sum(axis=-1) > 1e-7
    if not np.any(vis):
        return out
    counts = np.maximum(vis.sum(axis=1, keepdims=True), 1).astype(np.float32)
    center = (out * vis[..., None]).sum(axis=1, keepdims=True) / counts[:, None]
    has = vis.any(axis=1)
    center[~has] = 0.0
    out = out - center
    out = np.where(vis[..., None], out, 0.0)
    norms = np.linalg.norm(out, axis=-1)
    scale = float(norms[vis].mean()) if np.any(vis) else 0.0
    if scale > 1e-8:
        out = out / scale
        out = np.where(vis[..., None], out, 0.0)
    return out.astype(np.float32)


def add_velocity_v2(poses: np.ndarray) -> np.ndarray:
    """Velocity only between consecutive valid (non-zero) frames; else 0."""
    poses = poses.astype(np.float32)
    vel = np.zeros_like(poses)
    frame_energy = np.abs(poses).sum(axis=(1, 2))
    valid = frame_energy > 1e-7
    for t in range(1, len(poses)):
        if valid[t] and valid[t - 1]:
            vel[t] = poses[t] - poses[t - 1]
    return np.concatenate([poses, vel], axis=-1)


def extract_window(
    poses: np.ndarray,
    valid_mask: np.ndarray | None,
    start: int,
    end: int,
    *,
    window_len: int,
    pad_mode: str = "repeat_last",
    mask_policy: str = "keep_grid",
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    poses: [T, J, 2]
    Devuelve [window_len, J, 2] y máscara alineada si existe.
    """
    poses = np.nan_to_num(poses, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    vm = None
    if valid_mask is not None:
        vm = valid_mask.astype(bool).copy()

    chunk = poses[start:end]
    if vm is not None:
        chunk_m = vm[start:end]
    else:
        chunk_m = None

    if mask_policy == "filter_in_window" and chunk_m is not None:
        chunk = chunk[chunk_m]
        chunk_m = np.ones(len(chunk), dtype=bool) if len(chunk) else chunk_m

    t = len(chunk)
    if t < window_len:
        pad_n = window_len - t
        if pad_mode == "zeros":
            pad = np.zeros((pad_n,) + chunk.shape[1:], dtype=np.float32)
        else:
            pad = np.repeat(chunk[-1:], pad_n, axis=0) if t > 0 else np.zeros(
                (pad_n,) + chunk.shape[1:], dtype=np.float32
            )
        chunk = np.concatenate([chunk, pad], axis=0)
        if chunk_m is not None:
            pad_m = np.zeros(pad_n, dtype=bool)
            chunk_m = np.concatenate([chunk_m, pad_m])
    elif t > window_len:
        chunk = chunk[:window_len]
        if chunk_m is not None:
            chunk_m = chunk_m[:window_len]

    pad_frac = 0.0 if t >= window_len else (window_len - t) / float(window_len)

    if mask_policy == "keep_grid_v2":
        chunk = apply_keep_grid_v2(chunk, chunk_m)

    return chunk, chunk_m, pad_frac