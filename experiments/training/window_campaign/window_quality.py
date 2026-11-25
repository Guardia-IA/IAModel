"""Calidad por chunk: duración mínima + esqueleto (teletransporte, visibilidad, estático raro)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from window_index import WindowExample


@dataclass
class WindowQualityResult:
    ok: bool
    reasons: List[str]
    metrics: Dict[str, float]


def _visible_count(poses: np.ndarray, t: int) -> int:
    if t < 0 or t >= len(poses):
        return 0
    fr = poses[t]
    fin = np.isfinite(fr).all(axis=-1)
    nz = np.abs(fr).sum(axis=-1) > 1e-7
    return int((fin & nz).sum())


def _frame_max_joint_jump(poses: np.ndarray, t: int) -> float:
    if t + 1 >= len(poses):
        return 0.0
    d = poses[t + 1] - poses[t]
    return float(np.nanmax(np.linalg.norm(d, axis=-1)))


def _centroid(poses: np.ndarray, t: int) -> np.ndarray:
    fr = poses[t]
    m = np.isfinite(fr).all(axis=-1) & (np.abs(fr).sum(axis=-1) > 1e-7)
    if not np.any(m):
        return np.zeros(2, dtype=np.float32)
    return np.nanmean(fr[m], axis=0)


def assess_window_poses(
    poses: np.ndarray,
    valid_mask: np.ndarray | None,
    *,
    content_len: int,
    pad_frac: float,
    hz: float = 12.0,
    cfg: Optional[Dict[str, Any]] = None,
) -> WindowQualityResult:
    """poses: [T, J, 2] ventana ya recortada (con padding si aplica)."""
    cfg = cfg or {}
    min_sec = float(cfg.get("min_content_seconds", 2.0))
    min_pad_sec = float(cfg.get("min_padded_content_seconds", 1.0))
    allow_padded_tail = bool(cfg.get("allow_padded_tail", True))
    max_jump_ratio = float(cfg.get("max_jump_to_median_ratio", 8.0))
    min_vis_j = int(cfg.get("min_visible_joints", 5))
    max_low_vis = float(cfg.get("max_low_visibility_frame_ratio", 0.35))
    max_centroid = float(cfg.get("max_centroid_jump_norm", 0.55))
    max_zero_frac = float(cfg.get("max_all_zero_frame_ratio", 0.2))
    min_motion = float(cfg.get("min_median_motion_norm", 0.0))

    reasons: List[str] = []
    content_len = max(0, min(int(content_len), len(poses)))
    content_sec = content_len / hz if hz > 0 else 0.0

    if content_len <= 0:
        return WindowQualityResult(False, ["empty_window"], {"content_sec": 0.0})

    if pad_frac > 0.01 and allow_padded_tail:
        need = min_pad_sec
    else:
        need = min_sec
    if content_sec < need:
        reasons.append(f"short_content_{content_sec:.2f}s<{need:.2f}s")

    chunk = poses[:content_len]
    jumps = np.array([_frame_max_joint_jump(chunk, t) for t in range(max(0, content_len - 1))])
    med = float(np.median(jumps)) if len(jumps) else 0.0
    mx = float(np.max(jumps)) if len(jumps) else 0.0
    ratio = mx / (med + 1e-8)

    low_vis = sum(1 for t in range(content_len) if _visible_count(chunk, t) < min_vis_j)
    low_vis_frac = low_vis / max(1, content_len)

    zero_frames = sum(1 for t in range(content_len) if _visible_count(chunk, t) == 0)
    zero_frac = zero_frames / max(1, content_len)

    cent_jumps = []
    for t in range(max(0, content_len - 1)):
        c0 = _centroid(chunk, t)
        c1 = _centroid(chunk, t + 1)
        cent_jumps.append(float(np.linalg.norm(c1 - c0)))
    max_cent = max(cent_jumps) if cent_jumps else 0.0

    if ratio > max_jump_ratio and mx > 0.02:
        reasons.append(f"teleport_jump_ratio={ratio:.1f}>{max_jump_ratio}")
    if max_cent > max_centroid:
        reasons.append(f"centroid_jump={max_cent:.3f}>{max_centroid}")
    if low_vis_frac > max_low_vis:
        reasons.append(f"low_visibility_frac={low_vis_frac:.2f}")
    if zero_frac > max_zero_frac:
        reasons.append(f"zero_pose_frac={zero_frac:.2f}")
    if min_motion > 0 and med < min_motion:
        reasons.append(f"static_median_motion={med:.4f}")

    if valid_mask is not None and len(valid_mask) >= content_len:
        vm = valid_mask[:content_len].astype(bool)
        if not vm.all() and vm.sum() >= 2:
            filt = chunk[vm]
            fj = np.array(
                [_frame_max_joint_jump(filt, t) for t in range(max(0, len(filt) - 1))]
            )
            if len(fj):
                fm = float(np.median(fj))
                fx = float(np.max(fj))
                if fm > 1e-8 and fx > 3.0 * fm and fx > 0.03:
                    reasons.append("teleport_after_mask_filter")

    metrics = {
        "content_sec": content_sec,
        "pad_frac": float(pad_frac),
        "jump_median": med,
        "jump_max": mx,
        "jump_ratio": ratio,
        "low_vis_frac": low_vis_frac,
        "zero_frac": zero_frac,
        "centroid_jump_max": max_cent,
    }
    return WindowQualityResult(len(reasons) == 0, reasons, metrics)


def filter_windows_by_pose_quality(
    windows: List[WindowExample],
    *,
    cfg: Dict[str, Any],
    verbose: bool = True,
) -> Tuple[List[WindowExample], Dict[str, Any]]:
    if not (cfg.get("quality_gate") or {}).get("enabled", True):
        return windows, {"dropped": 0}

    qcfg = dict(cfg.get("quality_gate") or {})
    cache_poses: Dict[str, np.ndarray] = {}
    cache_vm: Dict[str, Optional[np.ndarray]] = {}
    kept: List[WindowExample] = []
    dropped = 0
    reason_counts: Dict[str, int] = {}

    for w in windows:
        key = str(w.pose_path.resolve())
        if key not in cache_poses:
            cache_poses[key] = np.load(w.pose_path)
            if w.valid_mask_path and w.valid_mask_path.exists():
                cache_vm[key] = np.load(w.valid_mask_path)
            else:
                cache_vm[key] = None
        full = cache_poses[key]
        vm = cache_vm[key]
        est = w.effective_start if w.effective_start is not None else w.start
        eend = w.effective_end if w.effective_end is not None else w.end
        slice_p = full[est:eend]
        slice_m = vm[est:eend] if vm is not None and len(vm) >= eend else None
        content_len = eend - est
        wl = int(cfg.get("window_len", 96))
        pad_frac = max(0.0, 1.0 - content_len / wl) if wl > 0 else 0.0

        # Padding sintético en el array para assess (solo contenido real importa)
        res = assess_window_poses(
            slice_p if len(slice_p) else full[0:0],
            slice_m,
            content_len=content_len,
            pad_frac=pad_frac,
            hz=float(qcfg.get("hz", 12.0)),
            cfg=qcfg,
        )
        if res.ok:
            kept.append(w)
        else:
            dropped += 1
            tag = res.reasons[0].split("=")[0] if res.reasons else "unknown"
            reason_counts[tag] = reason_counts.get(tag, 0) + 1

    summary = {"dropped": dropped, "kept": len(kept), "reason_counts": reason_counts}
    if verbose and dropped:
        print(
            f"[QUALITY] Ventanas descartadas={dropped} | conservadas={len(kept)} | "
            f"motivos={reason_counts}"
        )
    return kept, summary