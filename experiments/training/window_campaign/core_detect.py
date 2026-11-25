"""B1 — detect gesture core interval in staged clips (wrist speed)."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def _smooth_1d(x: np.ndarray, k: int = 5) -> np.ndarray:
    if len(x) < k:
        return x
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(x, kernel, mode="same")


def _merge_runs(mask: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    segs: list[tuple[int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        segs.append((i, j))
        i = j
    if not segs:
        return []
    merged = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = merged[-1]
        if s - pe <= max_gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged


def _wrist_indices(j: int) -> tuple[int, int]:
    if j >= 17:
        return 9, 10
    if j >= 8:
        return 4, 5
    return max(0, j - 2), max(0, j - 1)


def detect_core(
    poses: np.ndarray,
    *,
    hz: float = 12.0,
    dilate_frames: int = 6,
    trim_fallback: int = 12,
) -> Dict[str, Any]:
    """
    poses [T, J, 2]. Returns core_start, core_end (exclusive), confidence, method.
    """
    poses = np.asarray(poses, dtype=np.float32)
    t, j, _ = poses.shape
    if t < 2:
        return {
            "core_start": 0,
            "core_end": t,
            "confidence": 0.0,
            "method": "too_short",
        }

    li, ri = _wrist_indices(j)
    wrists = poses[:, [li, ri], :]
    diff = np.diff(wrists, axis=0)
    speed = np.nanmax(np.linalg.norm(diff, axis=-1), axis=1) * hz
    speed = np.nan_to_num(speed, nan=0.0)
    if speed.size == 0:
        speed = np.zeros(max(1, t - 1), dtype=np.float32)

    speed_s = _smooth_1d(speed, k=5)
    p95 = float(np.nanpercentile(speed_s, 95)) if np.any(speed_s > 0) else 0.0
    med = float(np.nanmedian(speed_s))
    thresh = 0.25 * p95 if p95 > 1e-8 else 0.0
    active = speed_s > thresh if thresh > 0 else speed_s > med

    # pad active to length T (velocity defined on T-1)
    active_t = np.zeros(t, dtype=bool)
    active_t[: len(active)] = active
    if len(active) > 0:
        active_t[len(active) :] = active[-1]

    segs = _merge_runs(active_t, max_gap=6)
    if not segs:
        cs, ce = trim_fallback, max(trim_fallback, t - trim_fallback)
        if ce <= cs:
            cs, ce = 0, t
        return {
            "core_start": int(cs),
            "core_end": int(ce),
            "confidence": 0.2,
            "method": "trim_fallback",
        }

    def seg_energy(seg: tuple[int, int]) -> float:
        s, e = seg
        return float(np.nansum(speed_s[s : min(e, len(speed_s))]))

    core_s, core_e = max(segs, key=seg_energy)
    core_s = max(0, core_s - dilate_frames)
    core_e = min(t, core_e + dilate_frames)
    conf = float(p95 / (med + 1e-6)) if med > 0 else 0.5
    conf = min(10.0, conf) / 10.0

    return {
        "core_start": int(core_s),
        "core_end": int(core_e),
        "confidence": conf,
        "method": "wrist_speed",
    }


def core_for_clip_file(path) -> Dict[str, Any]:
    poses = np.load(path)
    return detect_core(poses)