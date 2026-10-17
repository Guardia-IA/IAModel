"""Auditoría fps/meta vs longitud del npy (experto: tiempo coherente antes de 12 Hz)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from dataset_cache import _fps_from_meta


def _npy_num_frames(pose_path: str) -> int:
    p = Path(pose_path)
    if not p.is_file():
        return 0
    arr = np.load(p, mmap_mode="r")
    return int(arr.shape[0])


def audit_clip_timing(
    records: List[Dict[str, Any]],
    *,
    tolerance_sec: float = 0.35,
    max_samples: int = 25,
) -> Dict[str, Any]:
    """
    Compara duración implícita (T, fps meta) con clip_duration en meta.json si existe.
    """
    issues: List[Dict[str, Any]] = []
    checked = 0
    for rec in records:
        pose_path = str(rec.get("pose_path") or "")
        t = _npy_num_frames(pose_path)
        if t <= 0:
            continue
        checked += 1
        meta = rec.get("meta") or {}
        fps = float(_fps_from_meta(meta))
        implied = (t - 1) / fps if t > 1 and fps > 0 else 0.0
        clip_dur = meta.get("clip_duration") or meta.get("duration")
        if clip_dur is None:
            continue
        try:
            expected = float(clip_dur)
        except (TypeError, ValueError):
            continue
        delta = abs(implied - expected)
        if delta > tolerance_sec:
            issues.append(
                {
                    "uid": rec.get("uid"),
                    "clip_dir": rec.get("clip_dir"),
                    "frames": t,
                    "fps_meta": fps,
                    "duration_implied_sec": round(implied, 3),
                    "clip_duration_meta": expected,
                    "delta_sec": round(delta, 3),
                }
            )
    issues.sort(key=lambda x: -float(x.get("delta_sec", 0)))
    return {
        "clips_checked": checked,
        "n_issues": len(issues),
        "tolerance_sec": tolerance_sec,
        "issues_sample": issues[:max_samples],
    }
