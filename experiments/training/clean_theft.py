"""
Recorta un .npy de poses para quedarse solo con la ventana del robo.

Detecta tres firmas complementarias (normalizadas por altura del torso):
  1) reach   — extender el brazo y acercarlo al cuerpo (estantería, sin parar).
  2) conceal — acercar la muñeca a cadera/torso (meter en bolsillo caminando).
  3) low_pick — alcance hacia abajo (objeto en carrito bajo la cintura).

Agrupa picos reach+conceal en eventos de robo. Si hay varios en el mismo clip,
usa --all-events para generar un .npy por robo o --event-index N para elegir uno.

Formato soportado: [T, 8, 2] o [T, 2, 8, 2] (KEEP_KPS de pose_extractor_clean).
Índices locales: 0/1 hombros, 4/5 muñecas, 6/7 caderas. Coordenadas en [0, 1].

Uso:
    python clean_theft.py input.npy output.npy
    python clean_theft.py input.npy output.npy --all-events
    python clean_theft.py input.npy output.npy --event-index 1 --inspect
    python clean_theft.py input.npy output.npy --plot distancia.png
    python clean_theft.py input.npy output.npy --padding 8 --min-frames 20 --max-frames 90
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Índices locales sobre KEEP_KPS = [5,6,7,8,9,10,11,12]
LEFT_SHOULDER, RIGHT_SHOULDER = 0, 1
LEFT_WRIST, RIGHT_WRIST = 4, 5
LEFT_HIP, RIGHT_HIP = 6, 7

THEFT_MODES = ("auto", "reach", "conceal", "low_pick", "combined")


@dataclass
class TheftSignals:
    reach: np.ndarray
    conceal: np.ndarray
    low_pick: np.ndarray
    combined: np.ndarray


@dataclass
class TheftSegment:
    start: int
    end: int
    peak: int
    confidence: float
    reason: str
    mode: str = "combined"
    event_index: int = 0
    reach_peak: Optional[int] = None
    conceal_peak: Optional[int] = None

    @property
    def length(self) -> int:
        return self.end - self.start


def _kp_visible(kp: np.ndarray) -> bool:
    if kp.shape[-1] < 2:
        return False
    if np.isnan(kp).any():
        return False
    return not (float(kp[0]) == 0.0 and float(kp[1]) == 0.0)


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def _load_pose_sequence(npy_path: str, user_index: int = 0) -> Tuple[np.ndarray, bool]:
    data = np.load(npy_path)
    if data.ndim == 3:
        return data.astype(np.float64), False
    if data.ndim == 4 and data.shape[1] >= 1:
        idx = max(0, min(user_index, int(data.shape[1]) - 1))
        return data[:, idx, :, :2].astype(np.float64), True
    raise ValueError(
        f"Formato no soportado: {data.shape}. Esperado (T,8,2) o (T,U,8,2)."
    )


def _interpolate_nans(signal: np.ndarray) -> np.ndarray:
    out = signal.copy()
    valid = np.isfinite(out)
    if valid.all():
        return out
    if not valid.any():
        return np.zeros_like(out)
    idx = np.arange(len(out))
    out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])
    return out


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(signal) < window:
        return signal.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(signal, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _torso_refs(frame: np.ndarray) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    shoulders = []
    hips = []
    if _kp_visible(frame[LEFT_SHOULDER]):
        shoulders.append(frame[LEFT_SHOULDER])
    if _kp_visible(frame[RIGHT_SHOULDER]):
        shoulders.append(frame[RIGHT_SHOULDER])
    if _kp_visible(frame[LEFT_HIP]):
        hips.append(frame[LEFT_HIP])
    if _kp_visible(frame[RIGHT_HIP]):
        hips.append(frame[RIGHT_HIP])
    if not shoulders or not hips:
        return None
    shoulder_mid = np.mean(shoulders, axis=0)
    hip_mid = np.mean(hips, axis=0)
    torso_h = _euclidean(shoulder_mid, hip_mid)
    if torso_h <= 1e-6:
        return None
    torso_mid = 0.5 * (shoulder_mid + hip_mid)
    return torso_h, hip_mid, torso_mid


def _arm_side_values(
    frame: np.ndarray,
    torso_h: float,
    hip_mid: np.ndarray,
    torso_mid: np.ndarray,
    shoulder_idx: int,
    wrist_idx: int,
    hip_idx: int,
) -> Optional[Tuple[float, float, float]]:
    if not _kp_visible(frame[shoulder_idx]) or not _kp_visible(frame[wrist_idx]):
        return None
    wrist = frame[wrist_idx]
    shoulder = frame[shoulder_idx]

    reach = _euclidean(shoulder, wrist) / torso_h

    hip_ref = frame[hip_idx] if _kp_visible(frame[hip_idx]) else hip_mid
    dist_hip = _euclidean(wrist, hip_ref)
    dist_torso = _euclidean(wrist, torso_mid)
    body_dist = min(dist_hip, dist_torso)
    conceal = torso_h / (body_dist + 1e-6)

    below_waist = max(0.0, float(wrist[1] - hip_ref[1])) / torso_h
    low_pick = reach * (1.0 + 1.75 * below_waist)

    return reach, conceal, low_pick


def compute_theft_signals(poses: np.ndarray) -> TheftSignals:
    """
    reach:    pico al extender el brazo (coger en estantería, también en marcha).
    conceal:  pico al acercar la mano a cadera/torso (bolsillo).
    low_pick: pico al extender hacia abajo (carrito bajo cintura).
    """
    t_total = len(poses)
    reach = np.full(t_total, np.nan, dtype=np.float64)
    conceal = np.full(t_total, np.nan, dtype=np.float64)
    low_pick = np.full(t_total, np.nan, dtype=np.float64)

    for t in range(t_total):
        refs = _torso_refs(poses[t])
        if refs is None:
            continue
        torso_h, hip_mid, torso_mid = refs

        reaches, conceals, lows = [], [], []
        for shoulder_idx, wrist_idx, hip_idx in (
            (LEFT_SHOULDER, LEFT_WRIST, LEFT_HIP),
            (RIGHT_SHOULDER, RIGHT_WRIST, RIGHT_HIP),
        ):
            side = _arm_side_values(
                poses[t], torso_h, hip_mid, torso_mid, shoulder_idx, wrist_idx, hip_idx
            )
            if side is None:
                continue
            reaches.append(side[0])
            conceals.append(side[1])
            lows.append(side[2])

        if reaches:
            reach[t] = max(reaches)
            conceal[t] = max(conceals)
            low_pick[t] = max(lows)

    filled_reach = _interpolate_nans(reach)
    filled_conceal = _interpolate_nans(conceal)
    filled_low = _interpolate_nans(low_pick)
    combined = np.maximum(np.maximum(filled_reach, filled_conceal), filled_low)

    return TheftSignals(
        reach=filled_reach,
        conceal=filled_conceal,
        low_pick=filled_low,
        combined=combined,
    )


def compute_reach_signal(poses: np.ndarray) -> np.ndarray:
    """Compatibilidad: solo señal de alcance muñeca-hombro."""
    return compute_theft_signals(poses).reach


def _smooth_signals(signals: TheftSignals, smooth_window: int) -> Dict[str, np.ndarray]:
    return {
        "reach": _moving_average(signals.reach, smooth_window),
        "conceal": _moving_average(signals.conceal, smooth_window),
        "low_pick": _moving_average(signals.low_pick, smooth_window),
    }


def _baseline_from_edges(signal: np.ndarray, edge_frac: float = 0.18) -> float:
    """Baseline solo con inicio/fin del clip (suele ser caminar, no robo)."""
    n = len(signal)
    if n < 8:
        return float(np.median(signal))
    k = max(2, int(n * edge_frac))
    edges = np.concatenate([signal[:k], signal[n - k :]])
    return float(np.median(edges))


def _robust_amplitude(signal: np.ndarray) -> float:
    p10, p90 = np.percentile(signal, [10, 90])
    return max(float(p90 - p10), 1e-6)


def _find_local_maxima(signal: np.ndarray) -> list[int]:
    peaks: list[int] = []
    for i in range(1, len(signal) - 1):
        if signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1]:
            peaks.append(i)
    return peaks


def _peak_prominence(signal: np.ndarray, peak: int, width: int = 18) -> float:
    left = signal[max(0, peak - width) : peak]
    right = signal[peak + 1 : min(len(signal), peak + width + 1)]
    left_min = float(np.min(left)) if left.size else float(signal[peak])
    right_min = float(np.min(right)) if right.size else float(signal[peak])
    return max(0.0, float(signal[peak] - max(left_min, right_min)))


def _nms_peaks(candidates: list[Tuple[int, float]], min_distance: int) -> list[Tuple[int, float]]:
    selected: list[Tuple[int, float]] = []
    for idx, prom in sorted(candidates, key=lambda item: -item[1]):
        if all(abs(idx - kept) >= min_distance for kept, _ in selected):
            selected.append((idx, prom))
    return selected


def _edge_weight(frame: int, n: int, margin_frac: float = 0.14) -> float:
    margin = max(3, int(n * margin_frac))
    if frame < margin:
        return 0.12 + 0.88 * (frame / margin)
    if frame >= n - margin:
        return 0.12 + 0.88 * ((n - 1 - frame) / margin)
    return 1.0


def _rhythm_penalty(signal: np.ndarray, peak: int, window: int = 45) -> float:
    """
    Penaliza picos que forman parte de un patrón rítmico (balanceo al caminar).
    """
    left = max(0, peak - window)
    right = min(len(signal), peak + window + 1)
    local = signal[left:right]
    if len(local) < 24:
        return 1.0

    local_peaks = _find_local_maxima(local)
    min_prom = _robust_amplitude(local) * 0.08
    heights = [
        float(local[p])
        for p in local_peaks
        if _peak_prominence(local, p, width=12) >= min_prom
    ]
    if len(heights) < 3:
        return 1.0

    heights_arr = np.asarray(heights, dtype=np.float64)
    cv = float(np.std(heights_arr) / (np.mean(heights_arr) + 1e-9))
    if cv < 0.38:
        return 0.18
    return 1.0


def _collect_peak_candidates(
    smoothed: Dict[str, np.ndarray],
    mode: str,
) -> list[Tuple[int, str, float, float]]:
    """
    Devuelve candidatos (frame, modo, score, prominencia).
    El score usa prominencia real (no min-max por clip) + penalizaciones.
    """
    mode_names = ["reach", "conceal", "low_pick"]
    if mode not in ("auto", "combined"):
        mode_names = [mode]

    candidates: list[Tuple[int, str, float, float]] = []
    n = len(next(iter(smoothed.values())))

    for name in mode_names:
        signal = smoothed[name]
        amp = _robust_amplitude(signal)
        if name == "conceal":
            min_prom = max(amp * 0.12, 1.0)
        else:
            min_prom = max(amp * 0.12, 0.04)

        raw_candidates: list[Tuple[int, float]] = []
        for peak in _find_local_maxima(signal):
            prom = _peak_prominence(signal, peak, width=18)
            if prom >= min_prom:
                raw_candidates.append((peak, prom))

        for peak, prom in _nms_peaks(raw_candidates, min_distance=10):
            score = prom * _edge_weight(peak, n)
            if name in ("reach", "low_pick"):
                score *= _rhythm_penalty(signal, peak)
            candidates.append((peak, name, score, prom))

    return candidates


def _expand_segment(
    peak: int,
    smoothed: Dict[str, np.ndarray],
    min_frames: int,
    max_frames: int,
    padding: int,
    active_threshold_ratio: float,
) -> Tuple[int, int]:
    n = len(smoothed["reach"])
    baselines = {name: _baseline_from_edges(sig) for name, sig in smoothed.items()}

    def frame_active(t: int) -> bool:
        for name, sig in smoothed.items():
            prom = max(float(sig[peak] - baselines[name]), 1e-9)
            if float(sig[t] - baselines[name]) >= active_threshold_ratio * prom:
                return True
        return False

    start = peak
    while start > 0 and frame_active(start - 1):
        start -= 1
    end = peak + 1
    while end < n and frame_active(end):
        end += 1

    start = max(0, start - padding)
    end = min(n, end + padding)
    return _enforce_length(start, end, peak, n, min_frames, max_frames)


def _enforce_length(
    start: int,
    end: int,
    peak: int,
    n: int,
    min_frames: int,
    max_frames: int,
) -> Tuple[int, int]:
    if end - start < min_frames:
        extra = min_frames - (end - start)
        start = max(0, start - extra // 2)
        end = min(n, start + min_frames)

    if end - start > max_frames:
        half = max_frames // 2
        start = max(0, peak - half)
        end = min(n, start + max_frames)
        if end - start < max_frames:
            start = max(0, end - max_frames)
    return start, end


def _reach_peak_before(
    conceal_peak: int,
    smoothed: Dict[str, np.ndarray],
    start_limit: int,
    end_limit: int,
    lookback: int = 55,
) -> Optional[int]:
    """Busca la extensión del brazo inmediatamente anterior al gesto de bolsillo."""
    reach = smoothed["reach"]
    lo = max(start_limit, conceal_peak - lookback)
    hi = min(end_limit, max(start_limit, conceal_peak - 1))
    if hi < lo:
        return None

    chunk = reach[lo : hi + 1]
    local_peak = int(np.argmax(chunk)) + lo
    min_prom = max(_robust_amplitude(reach) * 0.06, 0.03)
    if _peak_prominence(reach, local_peak, width=12) >= min_prom:
        return local_peak
    return None


def _find_valley_between(
    peak_a: int,
    peak_b: int,
    smoothed: Dict[str, np.ndarray],
) -> int:
    """Frame más quieto entre dos picos (suele separar dos robos)."""
    lo, hi = sorted((peak_a, peak_b))
    if hi <= lo:
        return lo
    activity = smoothed["conceal"][lo : hi + 1] + 0.20 * smoothed["reach"][lo : hi + 1]
    return lo + int(np.argmin(activity))


def _event_bounds_from_anchors(
    conceal_peaks: list[int],
    anchor_idx: int,
    n: int,
    smoothed: Dict[str, np.ndarray],
) -> Tuple[int, int]:
    """Delimita cada robo; deja margen tras el valle para incluir la retracción al cuerpo."""
    current = conceal_peaks[anchor_idx]

    if anchor_idx == 0:
        left = 0
    else:
        prev = conceal_peaks[anchor_idx - 1]
        valley = _find_valley_between(prev, current, smoothed)
        left = max(0, valley - max(6, (current - prev) // 6))

    if anchor_idx == len(conceal_peaks) - 1:
        right = n
    else:
        nxt = conceal_peaks[anchor_idx + 1]
        valley = _find_valley_between(current, nxt, smoothed)
        gap = nxt - current
        slack = max(8, int(gap * 0.38))
        right = min(n, valley + slack)

    return left, right


def _conceal_peak_in_window(
    smoothed: Dict[str, np.ndarray],
    start: int,
    end: int,
    hint: int,
) -> int:
    """Pico de bolsillo dentro de la ventana (puede ser posterior al reach)."""
    lo = max(0, start)
    hi = min(len(smoothed["conceal"]) - 1, end)
    if hi <= lo:
        return hint
    chunk = smoothed["conceal"][lo : hi + 1]
    return lo + int(np.argmax(chunk))


def _expand_theft_cycle_window(
    reach_peak: Optional[int],
    conceal_peak: int,
    group_start: int,
    smoothed: Dict[str, np.ndarray],
    left_bound: int,
    right_bound: int,
    n: int,
    min_frames: int,
    max_frames: int,
    padding: int,
) -> Tuple[int, int]:
    """
    Ventana completa del robo: extensión del brazo + acercar al cuerpo/bolsillo.
    """
    reach = smoothed["reach"]
    conceal = smoothed["conceal"]
    reach_base = _baseline_from_edges(reach)
    conceal_base = _baseline_from_edges(conceal)

    if reach_peak is None:
        reach_peak = max(left_bound, group_start)
        chunk = reach[max(left_bound, group_start - 5) : conceal_peak + 1]
        if chunk.size:
            reach_peak = max(left_bound, group_start - 5) + int(np.argmax(chunk))

    r_peak_val = float(reach[reach_peak])
    r_prom = max(r_peak_val - reach_base, 1e-9)
    r_start_thr = reach_base + 0.14 * r_prom

    start = reach_peak
    while start > left_bound and float(reach[start - 1]) >= r_start_thr:
        start -= 1
    start = max(left_bound, min(group_start, start) - padding)

    conceal_peak = _conceal_peak_in_window(smoothed, start, right_bound, conceal_peak)
    c_peak_val = float(conceal[conceal_peak])
    c_prom = max(c_peak_val - conceal_base, 1e-9)

    r_settle = reach_base + 0.18 * r_prom
    c_settle = conceal_base + 0.14 * c_prom

    action_start = min(reach_peak, conceal_peak)
    retraction_ref = max(reach_peak, conceal_peak)
    search_limit = min(right_bound, retraction_ref + 18)

    end = retraction_ref + 1
    idle = 0
    while end < search_limit:
        r_active = float(reach[end]) > r_settle
        c_active = float(conceal[end]) > c_settle
        if r_active or c_active:
            end += 1
            idle = 0
        else:
            idle += 1
            if idle >= 5:
                break
            end += 1

    search_hi = min(search_limit, action_start + 50)
    if search_hi > conceal_peak:
        tail = conceal[conceal_peak : search_hi + 1]
        if tail.size > 1:
            sec_peak = conceal_peak + int(np.argmax(tail))
            if float(conceal[sec_peak]) > c_settle:
                end = max(end, sec_peak + padding + 4)

    # Cola mínima tras el pico: tiempo para meter la mano en el cuerpo/bolsillo.
    pocket_tail = min(right_bound, retraction_ref + max(14, padding + 8))
    end = max(end, pocket_tail)

    short_post_end = min(right_bound, retraction_ref + 14)
    if short_post_end > retraction_ref + 1:
        short_post = reach[retraction_ref : short_post_end + 1]
        local_min_frame = retraction_ref + int(np.argmin(short_post))
        end = max(end, min(right_bound, local_min_frame + padding + 4))

    end = min(right_bound, end + padding)
    return _enforce_length(start, end, conceal_peak, n, min_frames, max_frames)


def _select_conceal_anchors(
    candidates: list[Tuple[int, str, float, float]],
    min_score_ratio: float = 0.08,
    min_distance: int = 18,
    absolute_min_score: float = 2.5,
) -> list[Tuple[int, str, float, float]]:
    conceal = [item for item in candidates if item[1] == "conceal"]
    if not conceal:
        return []

    best_score = max(item[2] for item in conceal)
    relative_threshold = max(best_score * min_score_ratio, absolute_min_score)
    strong = [item for item in conceal if item[2] >= relative_threshold]
    strong.sort(key=lambda item: -item[2])

    kept: list[Tuple[int, str, float, float]] = []
    for item in strong:
        if all(abs(item[0] - prev[0]) >= min_distance for prev in kept):
            kept.append(item)
    return sorted(kept, key=lambda item: item[0])


def _prune_conceal_anchors(
    anchors: list[Tuple[int, str, float, float]],
) -> list[Tuple[int, str, float, float]]:
    """Con 3+ picos, descarta gestos débiles que suelen ser ruido o caminar."""
    if len(anchors) <= 2:
        return anchors
    max_prom = max(item[3] for item in anchors)
    cutoff = max(max_prom * 0.50, 7.0)
    strong = [item for item in anchors if item[3] >= cutoff]
    return strong if strong else [max(anchors, key=lambda item: item[3])]


def _merge_same_theft_anchors(
    anchors: list[Tuple[int, str, float, float]],
    smoothed: Dict[str, np.ndarray],
    max_gap: int = 58,
) -> list[Tuple[int, int, str, float, float]]:
    """
    Une picos conceal cercanos del mismo robo.
    Devuelve (pico_representativo, inicio_grupo, modo, score, prominencia).
    """
    if len(anchors) <= 1:
        a = anchors[0]
        return [(a[0], a[0], a[1], a[2], a[3])]

    conceal = smoothed["conceal"]
    baseline = _baseline_from_edges(conceal)
    merged: list[Tuple[int, int, str, float, float]] = []
    group = [anchors[0]]

    def _flush(group_items: list[Tuple[int, str, float, float]]) -> None:
        best = max(group_items, key=lambda item: item[3])
        merged.append((best[0], group_items[0][0], best[1], best[2], best[3]))

    for next_anchor in anchors[1:]:
        prev = group[-1]
        gap = next_anchor[0] - prev[0]
        if gap > max_gap:
            _flush(group)
            group = [next_anchor]
            continue

        mid = conceal[prev[0] : next_anchor[0] + 1]
        valley = float(np.min(mid)) if mid.size else baseline
        peak_ref = max(prev[3], next_anchor[3])
        elevated_valley = valley > baseline + 0.10 * peak_ref
        if gap <= 28 or elevated_valley:
            group.append(next_anchor)
        else:
            _flush(group)
            group = [next_anchor]

    _flush(group)
    return merged


def _build_conceal_event(
    anchor: Tuple[int, int, str, float, float],
    anchor_idx: int,
    conceal_anchors: list[Tuple[int, int, str, float, float]],
    smoothed: Dict[str, np.ndarray],
    n: int,
    min_frames: int,
    max_frames: int,
    padding: int,
    active_threshold_ratio: float,
    event_index: int,
) -> TheftSegment:
    conceal_peak, group_start, _, score, prom = anchor
    conceal_frames = [item[0] for item in conceal_anchors]
    left_bound, right_bound = _event_bounds_from_anchors(conceal_frames, anchor_idx, n, smoothed)
    reach_peak = _reach_peak_before(
        conceal_peak,
        smoothed,
        max(left_bound, group_start - 5),
        right_bound,
    )

    start, end = _expand_theft_cycle_window(
        reach_peak=reach_peak,
        conceal_peak=conceal_peak,
        group_start=group_start,
        smoothed=smoothed,
        left_bound=left_bound,
        right_bound=right_bound,
        n=n,
        min_frames=min_frames,
        max_frames=max_frames,
        padding=padding,
    )

    if reach_peak is None:
        reach_peak = _reach_peak_before(conceal_peak, smoothed, left_bound, right_bound)

    final_conceal_peak = _conceal_peak_in_window(smoothed, start, end, conceal_peak)

    amp = _robust_amplitude(smoothed["conceal"])
    confidence = float(
        np.clip(score / max(prom, 1e-6), 0.0, 1.0) * np.clip(prom / max(amp, 1e-6), 0.0, 1.0)
    )
    if reach_peak is not None:
        reason = "ok_reach_conceal"
    else:
        reason = "ok_conceal"
    return TheftSegment(
        start=start,
        end=end,
        peak=final_conceal_peak,
        confidence=confidence,
        reason=reason,
        mode="conceal",
        event_index=event_index,
        reach_peak=reach_peak,
        conceal_peak=final_conceal_peak,
    )


def _build_reach_only_event(
    peak: int,
    peak_mode: str,
    score: float,
    prom: float,
    smoothed: Dict[str, np.ndarray],
    n: int,
    min_frames: int,
    max_frames: int,
    padding: int,
    active_threshold_ratio: float,
    event_index: int,
) -> TheftSegment:
    start, end = _expand_segment(
        peak,
        smoothed,
        min_frames=min_frames,
        max_frames=max_frames,
        padding=padding,
        active_threshold_ratio=active_threshold_ratio,
    )
    amp = _robust_amplitude(smoothed[peak_mode])
    confidence = float(
        np.clip(score / max(prom, 1e-6), 0.0, 1.0) * np.clip(prom / max(amp, 1e-6), 0.0, 1.0)
    )
    return TheftSegment(
        start=start,
        end=end,
        peak=peak,
        confidence=confidence,
        reason=f"ok_{peak_mode}",
        mode=peak_mode,
        event_index=event_index,
        reach_peak=peak if peak_mode == "reach" else None,
        conceal_peak=None,
    )


def detect_theft_segments(
    poses: np.ndarray,
    mode: str = "auto",
    smooth_window: int = 5,
    padding: int = 6,
    min_frames: int = 20,
    max_frames: int = 90,
    threshold_ratio: float = 0.35,
) -> List[TheftSegment]:
    n = len(poses)
    if n == 0:
        return [TheftSegment(0, 0, 0, 0.0, "vacio", mode=mode)]

    if n <= min_frames:
        return [
            TheftSegment(0, n, n // 2, 0.25, "clip_muy_corto_sin_recorte", mode=mode, event_index=0)
        ]

    signals = compute_theft_signals(poses)
    if not np.isfinite(signals.reach).any():
        return [
            TheftSegment(0, n, n // 2, 0.0, "sin_keypoints_validos", mode=mode, event_index=0)
        ]

    smoothed = _smooth_signals(signals, smooth_window)
    candidates = _collect_peak_candidates(smoothed, mode)
    events: List[TheftSegment] = []

    conceal_anchors = _merge_same_theft_anchors(
        _prune_conceal_anchors(_select_conceal_anchors(candidates)),
        smoothed,
    )
    for idx, anchor in enumerate(conceal_anchors):
        events.append(
            _build_conceal_event(
                anchor,
                idx,
                conceal_anchors,
                smoothed,
                n,
                min_frames,
                max_frames,
                padding,
                threshold_ratio,
                event_index=idx,
            )
        )

    if not events and candidates:
        peak, peak_mode, best_score, best_prom = max(candidates, key=lambda item: item[2])
        amp = _robust_amplitude(smoothed[peak_mode])
        if best_prom >= max(amp * 0.08, 0.03):
            events.append(
                _build_reach_only_event(
                    peak,
                    peak_mode,
                    best_score,
                    best_prom,
                    smoothed,
                    n,
                    min_frames,
                    max_frames,
                    padding,
                    threshold_ratio,
                    event_index=0,
                )
            )

    if not events:
        mid = n // 2
        half = min(max_frames // 2, max(min_frames // 2, n // 3))
        events.append(
            TheftSegment(
                max(0, mid - half),
                min(n, mid + half),
                mid,
                0.15,
                "sin_picos_usando_ventana_central",
                mode=mode if mode != "auto" else "reach",
                event_index=0,
            )
        )

    for idx, seg in enumerate(events):
        seg.event_index = idx
    return events


def detect_theft_segment(
    poses: np.ndarray,
    mode: str = "auto",
    smooth_window: int = 5,
    padding: int = 6,
    min_frames: int = 20,
    max_frames: int = 90,
    threshold_ratio: float = 0.35,
    event_index: int = 0,
    pick: str = "first",
) -> TheftSegment:
    events = detect_theft_segments(
        poses,
        mode=mode,
        smooth_window=smooth_window,
        padding=padding,
        min_frames=min_frames,
        max_frames=max_frames,
        threshold_ratio=threshold_ratio,
    )
    if not events:
        return TheftSegment(0, 0, 0, 0.0, "vacio", mode=mode)

    if pick == "best":
        return max(events, key=lambda seg: seg.confidence)
    if pick == "last":
        return events[-1]
    if 0 <= event_index < len(events):
        return events[event_index]
    return events[0]


def clean_theft_array(
    data: np.ndarray,
    user_index: int = 0,
    mode: str = "auto",
    smooth_window: int = 5,
    padding: int = 6,
    min_frames: int = 20,
    max_frames: int = 90,
    threshold_ratio: float = 0.35,
    event_index: int = 0,
    pick: str = "first",
) -> Tuple[np.ndarray, TheftSegment, np.ndarray]:
    if data.ndim == 3:
        poses = data
    elif data.ndim == 4:
        poses, _ = _load_pose_sequence_from_array(data, user_index)
    else:
        raise ValueError(f"Formato no soportado: {data.shape}")

    segment = detect_theft_segment(
        poses,
        mode=mode,
        smooth_window=smooth_window,
        padding=padding,
        min_frames=min_frames,
        max_frames=max_frames,
        threshold_ratio=threshold_ratio,
        event_index=event_index,
        pick=pick,
    )
    return data[segment.start : segment.end].copy(), segment, data


def _load_pose_sequence_from_array(data: np.ndarray, user_index: int) -> Tuple[np.ndarray, bool]:
    idx = max(0, min(user_index, int(data.shape[1]) - 1))
    return data[:, idx, :, :2].astype(np.float64), True


def _event_output_path(output_npy: str, event_index: int) -> str:
    base, ext = os.path.splitext(output_npy)
    if not ext:
        ext = ".npy"
    return f"{base}_evt{event_index + 1:02d}{ext}"


def save_plot(
    out_path: str,
    signals: TheftSignals,
    segments: List[TheftSegment],
    fps: float = 20.0,
    smooth_window: int = 5,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Instala matplotlib para usar --plot (pip install matplotlib).") from exc

    smoothed = _smooth_signals(signals, smooth_window)
    primary = segments[0] if segments else None
    active_name = primary.mode if primary else "reach"
    active = smoothed.get(active_name, smoothed["reach"])
    velocity = np.gradient(active)
    t = np.arange(len(signals.reach)) / fps

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for name, color in (
        ("reach", "#1f77b4"),
        ("conceal", "#d62728"),
        ("low_pick", "#2ca02c"),
    ):
        axes[0].plot(t, smoothed[name], label=name, color=color, linewidth=1.0, alpha=0.55)

    colors = ["#2ecc71", "#27ae60", "#1abc9c", "#16a085"]
    for idx, segment in enumerate(segments):
        color = colors[idx % len(colors)]
        axes[0].axvspan(
            segment.start / fps,
            segment.end / fps,
            alpha=0.18,
            color=color,
            label=f"evento {idx + 1}",
        )
        axes[0].axvline(segment.peak / fps, color=color, linestyle="--", linewidth=1.2)
        if segment.reach_peak is not None:
            axes[0].axvline(
                segment.reach_peak / fps,
                color=color,
                linestyle=":",
                linewidth=1.0,
            )

    title = "Detección"
    if primary:
        title = (
            f"Detección ({len(segments)} evento(s); "
            f"evt1 conf={primary.confidence:.2f}, {primary.reason})"
        )
    axes[0].set_ylabel("señal (suavizada)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(title)

    axes[1].plot(t, velocity, color="orange", label=f"velocidad ({active_name})")
    for idx, segment in enumerate(segments):
        color = colors[idx % len(colors)]
        axes[1].axvspan(segment.start / fps, segment.end / fps, alpha=0.18, color=color)
        axes[1].axvline(segment.peak / fps, color=color, linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("tiempo (s)")
    axes[1].set_ylabel("d(signal)/dt")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def clean_theft_file(
    input_npy: str,
    output_npy: str,
    user_index: int = 0,
    mode: str = "auto",
    smooth_window: int = 5,
    padding: int = 6,
    min_frames: int = 20,
    max_frames: int = 90,
    threshold_ratio: float = 0.35,
    plot_path: Optional[str] = None,
    meta_path: Optional[str] = None,
    fps: float = 20.0,
    event_index: int = 0,
    pick: str = "first",
    all_events: bool = False,
) -> List[TheftSegment]:
    data = np.load(input_npy)
    if data.ndim == 3:
        poses = data.astype(np.float64)
    elif data.ndim == 4:
        poses, _ = _load_pose_sequence_from_array(data, user_index)
    else:
        raise ValueError(f"Formato no soportado: {data.shape}")

    signals = compute_theft_signals(poses)
    segments = detect_theft_segments(
        poses,
        mode=mode,
        smooth_window=smooth_window,
        padding=padding,
        min_frames=min_frames,
        max_frames=max_frames,
        threshold_ratio=threshold_ratio,
    )

    if all_events:
        written: List[TheftSegment] = []
        for seg in segments:
            out_path = _event_output_path(output_npy, seg.event_index)
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            np.save(out_path, data[seg.start : seg.end].copy())
            written.append(seg)
    else:
        segment = detect_theft_segment(
            poses,
            mode=mode,
            smooth_window=smooth_window,
            padding=padding,
            min_frames=min_frames,
            max_frames=max_frames,
            threshold_ratio=threshold_ratio,
            event_index=event_index,
            pick=pick,
        )
        out_dir = os.path.dirname(output_npy)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.save(output_npy, data[segment.start : segment.end].copy())
        written = [segment]

    if plot_path:
        save_plot(plot_path, signals, segments, fps=fps, smooth_window=smooth_window)

    if meta_path:
        payload: Dict[str, object] = {
            "input_npy": input_npy,
            "input_frames": int(len(poses)),
            "events_detected": len(segments),
            "fps_assumed": fps,
            "events": [],
        }
        for seg in segments:
            event_payload = {
                "output_npy": (
                    _event_output_path(output_npy, seg.event_index)
                    if all_events
                    else output_npy
                ),
                "output_frames": int(seg.length),
                "segment": asdict(seg),
                "time_range_s": {
                    "start": seg.start / fps,
                    "end": seg.end / fps,
                    "peak": seg.peak / fps,
                    "reach_peak": (seg.reach_peak / fps if seg.reach_peak is not None else None),
                },
                "signals_at_peak": {
                    "reach": float(signals.reach[seg.peak]),
                    "conceal": float(signals.conceal[seg.peak]),
                    "low_pick": float(signals.low_pick[seg.peak]),
                },
            }
            payload["events"].append(event_payload)
        meta_dir = os.path.dirname(meta_path)
        if meta_dir:
            os.makedirs(meta_dir, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recorta un .npy de poses para quedarse solo con la ventana del robo."
    )
    parser.add_argument("input_npy", type=str, help="Ruta al .npy de entrada")
    parser.add_argument("output_npy", type=str, help="Ruta al .npy recortado de salida")
    parser.add_argument(
        "--mode",
        choices=THEFT_MODES,
        default="auto",
        help=(
            "Señal de detección: auto (elige el pico más saliente entre reach/conceal/low_pick), "
            "reach, conceal, low_pick o combined."
        ),
    )
    parser.add_argument(
        "--user-index",
        type=int,
        default=0,
        help="Índice de usuario si el array es (T, U, J, 2). Por defecto 0.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=6,
        help="Frames extra antes/después de la ventana detectada (default: 6).",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=20,
        help="Mínimo de frames en el recorte (default: 20, ~1 s a 20 fps).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=90,
        help="Máximo de frames en el recorte (default: 90, ~4.5 s a 20 fps).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Ventana de suavizado temporal (default: 5).",
    )
    parser.add_argument(
        "--threshold-ratio",
        type=float,
        default=0.35,
        help="Umbral relativo al pico para delimitar inicio/fin (default: 0.35).",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Guarda gráfico de diagnóstico (las 3 señales + velocidad).",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default=None,
        help="Guarda JSON con frames detectados, modo y confianza.",
    )
    parser.add_argument(
        "--all-events",
        action="store_true",
        help=(
            "Genera un .npy por cada robo detectado: output_evt01.npy, output_evt02.npy, ..."
        ),
    )
    parser.add_argument(
        "--event-index",
        type=int,
        default=0,
        help=(
            "Índice del evento a exportar cuando hay varios (0 = primer robo). "
            "Ignorado con --all-events."
        ),
    )
    parser.add_argument(
        "--pick",
        choices=("first", "best", "last"),
        default="first",
        help=(
            "Qué evento exportar si hay varios y no usas --all-events: "
            "first (cronológico, default), best (mayor confianza) o last."
        ),
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Lista candidatos y eventos detectados antes de recortar (debug).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="FPS asumido para el eje temporal del gráfico (default: 20).",
    )
    return parser.parse_args()


def _print_inspect(poses: np.ndarray, mode: str, smooth_window: int) -> None:
    signals = compute_theft_signals(poses)
    smoothed = _smooth_signals(signals, smooth_window)
    candidates = _collect_peak_candidates(smoothed, mode)
    print("Candidatos (frame, modo, score, prominencia):")
    for peak, peak_mode, score, prom in sorted(candidates, key=lambda item: -item[2])[:12]:
        print(f"  frame {peak:3d}  {peak_mode:8s}  score={score:7.3f}  prom={prom:7.3f}")

    events = detect_theft_segments(poses, mode=mode, smooth_window=smooth_window)
    print(f"\nEventos detectados: {len(events)}")
    for seg in events:
        reach_txt = f", reach={seg.reach_peak}" if seg.reach_peak is not None else ""
        print(
            f"  evt {seg.event_index + 1}: frames [{seg.start}, {seg.end}) "
            f"pico={seg.peak}{reach_txt} conf={seg.confidence:.2f} ({seg.reason})"
        )


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.input_npy):
        print(f"Archivo no encontrado: {args.input_npy}")
        raise SystemExit(1)

    if args.inspect:
        poses, _ = _load_pose_sequence(args.input_npy, user_index=args.user_index)
        _print_inspect(poses, args.mode, args.smooth_window)
        print()
    segments = clean_theft_file(
        input_npy=args.input_npy,
        output_npy=args.output_npy,
        user_index=args.user_index,
        mode=args.mode,
        smooth_window=args.smooth_window,
        padding=args.padding,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        threshold_ratio=args.threshold_ratio,
        plot_path=args.plot,
        meta_path=args.meta,
        fps=args.fps,
        event_index=args.event_index,
        pick=args.pick,
        all_events=args.all_events,
    )

    print(f"Entrada:  {args.input_npy}")
    print(f"Eventos:  {len(segments)} exportado(s)")
    for seg in segments:
        out_path = (
            _event_output_path(args.output_npy, seg.event_index)
            if args.all_events
            else args.output_npy
        )
        print(f"Salida:   {out_path}")
        print(
            f"  evt {seg.event_index + 1}: frames [{seg.start}, {seg.end}) "
            f"(pico={seg.peak}, len={seg.length}, confianza={seg.confidence:.2f})"
        )
        if seg.reach_peak is not None:
            print(f"  reach_peak={seg.reach_peak}, conceal_peak={seg.conceal_peak}")
        print(f"  modo: {seg.mode} ({seg.reason})")
    if args.plot:
        print(f"Gráfico:  {args.plot}")
    if args.meta:
        print(f"Meta:     {args.meta}")


if __name__ == "__main__":
    main()
