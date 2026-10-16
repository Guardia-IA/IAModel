"""Normalización v2 (experto): ancla torso + escala isotrópica robusta."""
from __future__ import annotations

import numpy as np

# Orden KEEP_KPS [5..12]: hombros 0,1; codos 2,3; muñecas 4,5; caderas 6,7.
HIP_JOINTS = (6, 7)
SHOULDER_JOINTS = (0, 1)


def normalize_torso_isotropic(
    poses: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    poses: [T, J, 2] coordenadas imagen normalizadas.
    Centra cada frame en la cadera (media de hips visibles) y escala isotrópicamente
    por la distancia media torso (hombros/caderas) cuando hay validez.
    """
    poses = np.asarray(poses, dtype=np.float32)
    t, j, _ = poses.shape
    out = poses.copy()
    if valid is None:
        valid = np.ones((t, j), dtype=np.float32)
    else:
        valid = np.asarray(valid, dtype=np.float32)

    for ti in range(t):
        v = valid[ti]
        hip_pts = []
        for hj in HIP_JOINTS:
            if hj < j and v[hj] > 0.5 and np.isfinite(out[ti, hj]).all():
                hip_pts.append(out[ti, hj])
        if hip_pts:
            center = np.mean(hip_pts, axis=0, keepdims=True)
        else:
            mask = v > 0.5
            if mask.any():
                center = out[ti, mask].mean(axis=0, keepdims=True)
            else:
                center = out[ti].mean(axis=0, keepdims=True)

        rel = out[ti] - center
        dists: list[float] = []
        for sj in SHOULDER_JOINTS + HIP_JOINTS:
            if sj < j and v[sj] > 0.5:
                dists.append(float(np.linalg.norm(rel[sj])))
        if not dists:
            dists = [float(np.linalg.norm(rel, axis=-1).mean())]
        scale = max(float(np.mean(dists)), eps)
        out[ti] = rel / scale
    return out
