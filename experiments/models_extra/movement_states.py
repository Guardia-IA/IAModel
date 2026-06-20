"""
Seguimiento resumido de movimientos de brazos (8 keypoints KEEP_KPS).

Sin orientación a robo: solo estados estables y eventos al cambiar.
Incluye detección de mano hacia bolsillo/cadera (conceal, como clean_theft).

Normalización por escala corporal (torso + hombros) para que personas lejanas
no disparen falsos positivos de bolsillo por ruido de keypoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

LS, RS = 0, 1
LE, RE = 2, 3
LW, RW = 4, 5
LH, RH = 6, 7

MIN_KP_CONF = 0.5
DEFAULT_STABLE_FRAMES = 10
POCKET_STABLE_FRAMES = 6
AWAY_STABLE_FRAMES = 6

# Escala de referencia: sujeto medio en primer plano (coords normalizadas 0–1)
REF_TORSO_H = 0.20
REF_SHOULDER_W = 0.17
MIN_TORSO_H = 0.055
MIN_POCKET_TORSO_H = 0.065


class BodyZone(str, Enum):
    UNKNOWN = ""
    TORSO = "torso"
    HIP = "bolsillo/cadera"
    HIP_BACK = "bolsillo trasero"
    SHOULDER = "hombro"
    LOW = "zona baja"


ZONE_LABELS_ES: Dict[BodyZone, str] = {
    BodyZone.UNKNOWN: "cuerpo",
    BodyZone.TORSO: "torso",
    BodyZone.HIP: "bolsillo/cadera",
    BodyZone.HIP_BACK: "bolsillo trasero",
    BodyZone.SHOULDER: "hombro",
    BodyZone.LOW: "zona baja",
}


class ArmSummary(str, Enum):
    NO_DATA = "sin_datos"
    CALM = "tranquilo"
    EXTENDED_NEAR = "extendido_cerca_cuerpo"
    EXTENDED_AWAY = "extendido_alejado"
    TOWARD_POCKET = "hacia_bolsillo"
    RETRACTED = "recogido"


# Umbrales base (referencia REF_BODY_SCALE); bolsillo usa versión adaptativa
BODY_NEAR_MAX = 0.38
BODY_AWAY_MIN = 0.44
BODY_AWAY_STRONG = 0.52
REACH_MIN = 0.44
HIP_SIDE_MAX = 0.40
CONCEAL_MIN = 2.2
CONCEAL_STRONG = 2.8
POCKET_REACH_MAX = 0.58
POCKET_ELBOW_MAX_DEG = 135.0
HIP_OVERLAP_MAX = 0.20
HIP_POCKET_TIGHT = 0.26
REST_ELBOW_MIN_DEG = 148.0
POCKET_SCORE_MIN = 4
POCKET_SCORE_OVERLAP = 3
WRIST_HIP_DY_MAX = 0.11
REST_WRIST_HIP_DY_MIN = 0.12


@dataclass(frozen=True)
class PocketThresholds:
    hip_side_max: float
    conceal_min: float
    conceal_strong: float
    require_elbow_bend: bool
    reach_max: float


@dataclass(frozen=True)
class PersonNorm:
    """Referencia de escala del sujeto en coords normalizadas (0–1)."""

    torso_h: float
    shoulder_w: float
    scale_ratio: float  # torso_h / REF_TORSO_H; <1 = sujeto más lejos/pequeño

    @property
    def reliable(self) -> bool:
        return self.torso_h >= MIN_TORSO_H

    @property
    def pocket_ok(self) -> bool:
        return self.torso_h >= MIN_POCKET_TORSO_H

    def pocket_thresholds(self) -> PocketThresholds:
        """Umbrales más estrictos cuanto más pequeño es el sujeto en imagen."""
        r = float(np.clip(self.scale_ratio, 0.45, 1.15))
        hip_side_max = HIP_SIDE_MAX * (0.58 + 0.42 * r)
        conceal_min = CONCEAL_MIN + (1.0 - r) * 1.15
        conceal_strong = CONCEAL_STRONG + (1.0 - r) * 0.85
        require_elbow_bend = r < 0.78
        reach_max = POCKET_REACH_MAX - (1.0 - r) * 0.06
        return PocketThresholds(
            hip_side_max=hip_side_max,
            conceal_min=conceal_min,
            conceal_strong=conceal_strong,
            require_elbow_bend=require_elbow_bend,
            reach_max=reach_max,
        )


@dataclass
class ArmFrameMetrics:
    reach: float = np.nan
    body_dist: float = np.nan
    hip_side_dist: float = np.nan
    conceal: float = np.nan
    low: float = np.nan
    elbow_flex_deg: float = np.nan
    wrist_hip_dy: float = np.nan
    pocket_score: int = 0
    reach_vel: float = 0.0
    body_dist_vel: float = 0.0
    conceal_vel: float = 0.0


@dataclass
class FrameMovementResult:
    left_summary: ArmSummary = ArmSummary.NO_DATA
    right_summary: ArmSummary = ArmSummary.NO_DATA
    left_zone: BodyZone = BodyZone.UNKNOWN
    right_zone: BodyZone = BodyZone.UNKNOWN
    left_metrics: ArmFrameMetrics = field(default_factory=ArmFrameMetrics)
    right_metrics: ArmFrameMetrics = field(default_factory=ArmFrameMetrics)
    person_norm: Optional[PersonNorm] = None
    events: List[str] = field(default_factory=list)
    status_line: str = ""


def _kp_visible(kp: np.ndarray, conf: float) -> bool:
    if conf <= MIN_KP_CONF:
        return False
    if kp.shape[-1] < 2 or np.isnan(kp).any():
        return False
    return not (float(kp[0]) == 0.0 and float(kp[1]) == 0.0)


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def _shoulder_width(frame: np.ndarray, confs: np.ndarray) -> Optional[float]:
    if not (_kp_visible(frame[LS], confs[LS]) and _kp_visible(frame[RS], confs[RS])):
        return None
    return _euclidean(frame[LS], frame[RS])


def _elbow_flex_deg(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> Optional[float]:
    v1 = shoulder[:2] - elbow[:2]
    v2 = wrist[:2] - elbow[:2]
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 <= 1e-8 or n2 <= 1e-8:
        return None
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def _person_norm(frame: np.ndarray, confs: np.ndarray) -> Optional[PersonNorm]:
    shoulders, hips = [], []
    for idx in (LS, RS):
        if _kp_visible(frame[idx], confs[idx]):
            shoulders.append(frame[idx])
    for idx in (LH, RH):
        if _kp_visible(frame[idx], confs[idx]):
            hips.append(frame[idx])
    if not shoulders or not hips:
        return None

    shoulder_mid = np.mean(shoulders, axis=0)
    hip_mid = np.mean(hips, axis=0)
    torso_h = _euclidean(shoulder_mid, hip_mid)
    if torso_h <= 1e-6:
        return None

    shoulder_w = _shoulder_width(frame, confs)
    if shoulder_w is None:
        shoulder_w = torso_h * (REF_SHOULDER_W / REF_TORSO_H)
    else:
        shoulder_w = max(shoulder_w, torso_h * 0.42)

    scale_ratio = torso_h / REF_TORSO_H
    return PersonNorm(
        torso_h=torso_h,
        shoulder_w=shoulder_w,
        scale_ratio=scale_ratio,
    )


def _torso_refs(
    frame: np.ndarray, confs: np.ndarray, norm: PersonNorm
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    shoulders, hips = [], []
    for idx in (LS, RS):
        if _kp_visible(frame[idx], confs[idx]):
            shoulders.append(frame[idx])
    for idx in (LH, RH):
        if _kp_visible(frame[idx], confs[idx]):
            hips.append(frame[idx])
    if not shoulders or not hips:
        return None
    shoulder_mid = np.mean(shoulders, axis=0)
    hip_mid = np.mean(hips, axis=0)
    torso_mid = 0.5 * (shoulder_mid + hip_mid)
    return hip_mid, torso_mid, shoulder_mid


def _is_likely_back_pocket(
    wrist: np.ndarray,
    shoulder: np.ndarray,
    hip: np.ndarray,
    torso_mid: np.ndarray,
    norm: PersonNorm,
    is_right: bool,
) -> bool:
    """Heurística 2D normalizada por escala corporal."""
    th = norm.torso_h
    wy, hy = float(wrist[1]), float(hip[1])
    if wy < hy - 0.12 * th or wy > hy + 0.22 * th:
        return False

    wx, hx, sx = float(wrist[0]), float(hip[0]), float(shoulder[0])
    tx = float(torso_mid[0])
    inward = 0.035 * th
    lateral = 0.10 * th
    mid_tol = 0.12 * th
    behind_tol = 0.05 * th

    if is_right:
        if wx < hx - inward and abs(wy - hy) <= 0.10 * th:
            return True
        crossed_inward = wx <= hx + lateral or wx <= tx + mid_tol
        behind_line = wx <= max(hx, sx) + behind_tol
    else:
        if wx > hx + inward and abs(wy - hy) <= 0.10 * th:
            return True
        crossed_inward = wx >= hx - lateral or wx >= tx - mid_tol
        behind_line = wx >= min(hx, sx) - behind_tol

    return crossed_inward and behind_line


def _pocket_zone(
    frame: np.ndarray,
    confs: np.ndarray,
    wrist_idx: int,
    shoulder_idx: int,
    hip_idx: int,
    norm: PersonNorm,
    hip_mid: np.ndarray,
    torso_mid: np.ndarray,
    is_right: bool,
) -> BodyZone:
    if not _kp_visible(frame[wrist_idx], confs[wrist_idx]):
        return BodyZone.UNKNOWN
    wrist = frame[wrist_idx]
    hip_ref = frame[hip_idx] if _kp_visible(frame[hip_idx], confs[hip_idx]) else hip_mid
    shoulder = frame[shoulder_idx] if _kp_visible(frame[shoulder_idx], confs[shoulder_idx]) else None

    if shoulder is not None and _is_likely_back_pocket(wrist, shoulder, hip_ref, torso_mid, norm, is_right):
        return BodyZone.HIP_BACK

    if float(wrist[1]) > float(hip_ref[1]) + 0.16 * norm.torso_h:
        return BodyZone.LOW

    return BodyZone.HIP


def _arm_metrics(
    frame: np.ndarray,
    confs: np.ndarray,
    shoulder_idx: int,
    elbow_idx: int,
    wrist_idx: int,
    hip_idx: int,
    norm: PersonNorm,
    hip_mid: np.ndarray,
    torso_mid: np.ndarray,
) -> Optional[ArmFrameMetrics]:
    if not _kp_visible(frame[shoulder_idx], confs[shoulder_idx]):
        return None
    if not _kp_visible(frame[wrist_idx], confs[wrist_idx]):
        return None

    bs = norm.torso_h
    wrist = frame[wrist_idx]
    shoulder = frame[shoulder_idx]
    hip_ref = frame[hip_idx] if _kp_visible(frame[hip_idx], confs[hip_idx]) else hip_mid

    reach = _euclidean(shoulder, wrist) / bs
    dist_hip = _euclidean(wrist, hip_ref)
    dist_torso = _euclidean(wrist, torso_mid)
    raw_dist = min(dist_hip, dist_torso)
    body_dist = raw_dist / bs
    hip_side_dist = dist_hip / bs
    conceal = bs / (raw_dist + 1e-6)

    below = max(0.0, float(wrist[1] - hip_ref[1])) / norm.torso_h
    low = reach * (1.0 + 1.75 * below)

    elbow_flex = np.nan
    if _kp_visible(frame[elbow_idx], confs[elbow_idx]):
        flex = _elbow_flex_deg(shoulder, frame[elbow_idx], wrist)
        if flex is not None:
            elbow_flex = flex

    wrist_hip_dy = abs(float(wrist[1] - hip_ref[1])) / norm.torso_h

    return ArmFrameMetrics(
        reach=reach,
        body_dist=body_dist,
        hip_side_dist=hip_side_dist,
        conceal=conceal,
        low=low,
        elbow_flex_deg=elbow_flex,
        wrist_hip_dy=wrist_hip_dy,
    )


def _wrist_inward_toward_body(
    wrist: np.ndarray,
    hip: np.ndarray,
    shoulder: np.ndarray,
    torso_mid: np.ndarray,
    norm: PersonNorm,
    is_right: bool,
) -> bool:
    """Muñeca cruzada hacia el torso respecto a la posición lateral en reposo."""
    th = norm.torso_h
    wx, hx, sx = float(wrist[0]), float(hip[0]), float(shoulder[0])
    tx = float(torso_mid[0])
    if is_right:
        return wx < hx - 0.025 * th or wx < tx + 0.06 * th
    return wx > hx + 0.025 * th or wx > tx - 0.06 * th


def _is_extended_away(metrics: ArmFrameMetrics) -> bool:
    """Brazo claramente alejado del cuerpo (coger en estantería, etc.)."""
    body = metrics.body_dist if np.isfinite(metrics.body_dist) else 0.0
    reach = metrics.reach if np.isfinite(metrics.reach) else 0.0
    if body >= BODY_AWAY_STRONG and reach >= 0.38:
        return True
    if body >= BODY_AWAY_MIN and reach >= REACH_MIN:
        return True
    if body >= 0.40 and reach >= 0.50:
        return True
    if metrics.low >= 1.22 and body >= 0.38:
        return True
    return False


def _is_arm_hanging_at_rest(
    frame: np.ndarray,
    confs: np.ndarray,
    shoulder_idx: int,
    elbow_idx: int,
    wrist_idx: int,
    hip_idx: int,
    norm: PersonNorm,
    torso_mid: np.ndarray,
    is_right: bool,
    metrics: Optional[ArmFrameMetrics] = None,
) -> bool:
    """
    Brazo colgando a lo largo del cuerpo (reposo / caminar).
    Típico falso positivo de bolsillo: muñeca cerca de cadera pero codo recto y lateral.
    """
    if metrics is not None:
        body = metrics.body_dist if np.isfinite(metrics.body_dist) else 0.0
        if body >= 0.40:
            return False

    if not all(
        _kp_visible(frame[i], confs[i])
        for i in (shoulder_idx, elbow_idx, wrist_idx, hip_idx)
    ):
        return False

    th = norm.torso_h
    shoulder = frame[shoulder_idx]
    elbow = frame[elbow_idx]
    wrist = frame[wrist_idx]
    hip = frame[hip_idx]

    flex = _elbow_flex_deg(shoulder, elbow, wrist)
    if flex is None or flex < REST_ELBOW_MIN_DEG:
        return False

    wx, hx = float(wrist[0]), float(hip[0])
    wy, hy, ey = float(wrist[1]), float(hip[1]), float(elbow[1])
    wrist_hip_dy = abs(wy - hy) / th

    # Reposo claro: codo recto y muñeca por debajo de la cadera
    if flex >= REST_ELBOW_MIN_DEG and wy > hy + 0.06 * th:
        return True
    if flex >= 152.0 and wrist_hip_dy >= REST_WRIST_HIP_DY_MIN:
        return True

    lateral = (is_right and wx >= hx - 0.04 * th) or (not is_right and wx <= hx + 0.04 * th)
    hanging_down = wy >= ey - 0.03 * th
    below_hip = wy > hy + 0.05 * th

    if flex >= 155.0 and lateral and hanging_down:
        return True
    if flex >= REST_ELBOW_MIN_DEG and lateral and below_hip:
        return True
    if (
        flex >= 160.0
        and metrics is not None
        and np.isfinite(metrics.body_dist)
        and metrics.body_dist <= BODY_NEAR_MAX
        and not _wrist_inward_toward_body(wrist, hip, shoulder, torso_mid, norm, is_right)
    ):
        return True
    return False


def _pocket_insertion_score(
    metrics: ArmFrameMetrics,
    frame: np.ndarray,
    confs: np.ndarray,
    shoulder_idx: int,
    wrist_idx: int,
    hip_idx: int,
    norm: PersonNorm,
    torso_mid: np.ndarray,
    is_right: bool,
) -> int:
    """Puntuación compuesta: bolsillo real exige proximidad + postura, no solo 'cerca del cuerpo'."""
    score = 0
    th = norm.torso_h
    hip_d = metrics.hip_side_dist

    if hip_d <= HIP_OVERLAP_MAX:
        score += 3
    elif hip_d <= HIP_POCKET_TIGHT:
        score += 2
    elif hip_d <= 0.30:
        score += 1

    flex = metrics.elbow_flex_deg
    if np.isfinite(flex):
        if flex <= 120.0:
            score += 2
        elif flex <= POCKET_ELBOW_MAX_DEG:
            score += 1

    if np.isfinite(metrics.wrist_hip_dy) and metrics.wrist_hip_dy <= 0.09:
        score += 1

    if _kp_visible(frame[wrist_idx], confs[wrist_idx]) and _kp_visible(frame[hip_idx], confs[hip_idx]):
        wrist = frame[wrist_idx]
        hip = frame[hip_idx]
        shoulder = frame[shoulder_idx] if _kp_visible(frame[shoulder_idx], confs[shoulder_idx]) else hip
        if _wrist_inward_toward_body(wrist, hip, shoulder, torso_mid, norm, is_right):
            score += 1
        if shoulder is not None and _is_likely_back_pocket(wrist, shoulder, hip, torso_mid, norm, is_right):
            score += 1

    if metrics.conceal >= 3.4:
        score += 2
    elif metrics.conceal >= 3.0:
        score += 1

    return score


def _is_toward_pocket(
    metrics: ArmFrameMetrics,
    prev: Optional[ArmFrameMetrics],
    norm: PersonNorm,
    thr: PocketThresholds,
    frame: np.ndarray,
    confs: np.ndarray,
    shoulder_idx: int,
    elbow_idx: int,
    wrist_idx: int,
    hip_idx: int,
    torso_mid: np.ndarray,
    is_right: bool,
) -> bool:
    if not norm.pocket_ok:
        return False
    if not np.isfinite(metrics.hip_side_dist) or not np.isfinite(metrics.conceal):
        return False

    if _is_arm_hanging_at_rest(
        frame, confs, shoulder_idx, elbow_idx, wrist_idx, hip_idx, norm, torso_mid, is_right, metrics
    ):
        return False

    dy = metrics.wrist_hip_dy
    if not np.isfinite(dy) or dy > WRIST_HIP_DY_MAX:
        return False

    flex = metrics.elbow_flex_deg
    hip_d = metrics.hip_side_dist
    # Brazo recto con muñeca aún por debajo de la cadera → reposo, no bolsillo
    if np.isfinite(flex) and flex > 148.0 and dy > 0.095:
        return False
    # Codo muy recto solo vale si la muñeca está pegada a la cadera (superposición)
    if np.isfinite(flex) and flex > 152.0:
        if hip_d > 0.17 or dy > 0.085:
            return False

    score = _pocket_insertion_score(
        metrics, frame, confs, shoulder_idx, wrist_idx, hip_idx, norm, torso_mid, is_right
    )
    metrics.pocket_score = score

    min_score = POCKET_SCORE_OVERLAP if metrics.hip_side_dist <= HIP_OVERLAP_MAX else POCKET_SCORE_MIN

    if score >= min_score:
        return True

    # Acercamiento activo hacia bolsillo: subida de conceal + codo doblado + ya cerca
    if (
        prev is not None
        and np.isfinite(prev.conceal)
        and metrics.conceal_vel > 0.35
        and score >= POCKET_SCORE_MIN - 1
        and np.isfinite(metrics.elbow_flex_deg)
        and metrics.elbow_flex_deg <= POCKET_ELBOW_MAX_DEG
        and metrics.hip_side_dist <= HIP_POCKET_TIGHT
    ):
        return True
    return False


def _raw_arm_summary(
    metrics: Optional[ArmFrameMetrics],
    prev: Optional[ArmFrameMetrics],
    frame: np.ndarray,
    confs: np.ndarray,
    shoulder_idx: int,
    elbow_idx: int,
    wrist_idx: int,
    hip_idx: int,
    norm: PersonNorm,
    hip_mid: np.ndarray,
    torso_mid: np.ndarray,
    is_right: bool,
) -> Tuple[ArmSummary, BodyZone]:
    if metrics is None or not np.isfinite(metrics.reach):
        return ArmSummary.NO_DATA, BodyZone.UNKNOWN

    thr = norm.pocket_thresholds()
    body_dist = metrics.body_dist if np.isfinite(metrics.body_dist) else 999.0
    reach = metrics.reach
    low = metrics.low if np.isfinite(metrics.low) else 0.0
    reach_vel = metrics.reach_vel
    body_dist_vel = metrics.body_dist_vel

    prev_away = (
        prev is not None
        and np.isfinite(prev.body_dist)
        and prev.body_dist >= BODY_AWAY_MIN
    )

    # Alejado del cuerpo tiene prioridad sobre reposo/bolsillo
    if _is_extended_away(metrics):
        return ArmSummary.EXTENDED_AWAY, BodyZone.TORSO

    if body_dist_vel > 0.012 and reach_vel > 0.010 and body_dist >= 0.38:
        return ArmSummary.EXTENDED_AWAY, BodyZone.TORSO

    if _is_arm_hanging_at_rest(
        frame, confs, shoulder_idx, elbow_idx, wrist_idx, hip_idx, norm, torso_mid, is_right, metrics
    ):
        return ArmSummary.CALM, BodyZone.TORSO

    if _is_toward_pocket(
        metrics, prev, norm, thr, frame, confs,
        shoulder_idx, elbow_idx, wrist_idx, hip_idx, torso_mid, is_right,
    ):
        zone = _pocket_zone(
            frame, confs, wrist_idx, shoulder_idx, hip_idx,
            norm, hip_mid, torso_mid, is_right,
        )
        return ArmSummary.TOWARD_POCKET, zone

    if low >= 1.22 and body_dist >= 0.38:
        return ArmSummary.EXTENDED_AWAY, BodyZone.LOW

    if prev_away and body_dist <= BODY_NEAR_MAX + 0.08:
        zone = _pocket_zone(
            frame, confs, wrist_idx, shoulder_idx, hip_idx,
            norm, hip_mid, torso_mid, is_right,
        )
        return ArmSummary.RETRACTED, zone

    if prev_away and body_dist_vel < -0.014:
        zone = _pocket_zone(
            frame, confs, wrist_idx, shoulder_idx, hip_idx,
            norm, hip_mid, torso_mid, is_right,
        )
        return ArmSummary.RETRACTED, zone

    if body_dist <= BODY_NEAR_MAX and reach >= 0.42:
        return ArmSummary.EXTENDED_NEAR, BodyZone.TORSO

    if body_dist <= BODY_NEAR_MAX and reach <= 0.58:
        return ArmSummary.CALM, BodyZone.TORSO

    if body_dist <= BODY_NEAR_MAX:
        return ArmSummary.EXTENDED_NEAR, BodyZone.TORSO

    return ArmSummary.CALM, BodyZone.TORSO


@dataclass
class _SideTracker:
    side_label: str
    stable: ArmSummary = ArmSummary.NO_DATA
    stable_zone: BodyZone = BodyZone.UNKNOWN
    _candidate: ArmSummary = ArmSummary.NO_DATA
    _candidate_zone: BodyZone = BodyZone.UNKNOWN
    _candidate_count: int = 0

    def _min_stable(self, raw: ArmSummary) -> int:
        if raw == ArmSummary.TOWARD_POCKET:
            return POCKET_STABLE_FRAMES
        if raw == ArmSummary.EXTENDED_AWAY:
            return AWAY_STABLE_FRAMES
        return DEFAULT_STABLE_FRAMES

    def update(self, raw: ArmSummary, zone: BodyZone) -> List[str]:
        if raw == self._candidate and zone == self._candidate_zone:
            self._candidate_count += 1
        else:
            self._candidate = raw
            self._candidate_zone = zone
            self._candidate_count = 1

        needed = self._min_stable(raw)
        if self._candidate_count < needed:
            return []

        if raw == self.stable and zone == self.stable_zone:
            return []

        self.stable = raw
        self.stable_zone = zone
        return self._event_text()

    def _event_text(self) -> List[str]:
        side = f"Brazo {self.side_label}"
        if self.stable == ArmSummary.TOWARD_POCKET:
            if self.stable_zone == BodyZone.HIP_BACK:
                return [f"{side} hacia bolsillo trasero"]
            if self.stable_zone == BodyZone.LOW:
                return [f"{side} hacia zona baja / bolsillo"]
            return [f"{side} hacia bolsillo o cadera"]
        if self.stable == ArmSummary.EXTENDED_AWAY:
            if self.stable_zone == BodyZone.LOW:
                return [f"{side} extendido alejado (hacia abajo)"]
            return [f"{side} extendido alejado del cuerpo"]
        if self.stable == ArmSummary.RETRACTED:
            where = ZONE_LABELS_ES.get(self.stable_zone, "cuerpo")
            return [f"{side} recogido → {where}"]
        return []


class MovementClassifier:
    def __init__(self, stable_frames: int = DEFAULT_STABLE_FRAMES) -> None:
        self.stable_frames = max(3, stable_frames)
        self._prev_left: Optional[ArmFrameMetrics] = None
        self._prev_right: Optional[ArmFrameMetrics] = None
        self._left = _SideTracker("izquierdo")
        self._right = _SideTracker("derecho")
        self._both_calm_logged = False
        self._ema_body_scale: Optional[float] = None

    def reset(self) -> None:
        self._prev_left = None
        self._prev_right = None
        self._left = _SideTracker("izquierdo")
        self._right = _SideTracker("derecho")
        self._both_calm_logged = False
        self._ema_body_scale = None

    def _smooth_norm(self, raw: PersonNorm) -> PersonNorm:
        if self._ema_body_scale is None:
            self._ema_body_scale = raw.torso_h
        else:
            self._ema_body_scale = 0.82 * self._ema_body_scale + 0.18 * raw.torso_h
        ratio = self._ema_body_scale / REF_TORSO_H
        return PersonNorm(
            torso_h=raw.torso_h,
            shoulder_w=raw.shoulder_w,
            scale_ratio=ratio,
        )

    def _with_vel(
        self,
        metrics: Optional[ArmFrameMetrics],
        prev: Optional[ArmFrameMetrics],
    ) -> Optional[ArmFrameMetrics]:
        if metrics is None:
            return None
        out = ArmFrameMetrics(
            reach=metrics.reach,
            body_dist=metrics.body_dist,
            hip_side_dist=metrics.hip_side_dist,
            conceal=metrics.conceal,
            low=metrics.low,
            elbow_flex_deg=metrics.elbow_flex_deg,
            wrist_hip_dy=metrics.wrist_hip_dy,
            pocket_score=metrics.pocket_score,
        )
        if prev is not None and np.isfinite(prev.reach) and np.isfinite(metrics.reach):
            out.reach_vel = metrics.reach - prev.reach
        if prev is not None and np.isfinite(prev.body_dist) and np.isfinite(metrics.body_dist):
            out.body_dist_vel = metrics.body_dist - prev.body_dist
        if prev is not None and np.isfinite(prev.conceal) and np.isfinite(metrics.conceal):
            out.conceal_vel = metrics.conceal - prev.conceal
        return out

    def classify(self, frame: np.ndarray, confs: np.ndarray) -> FrameMovementResult:
        raw_norm = _person_norm(frame, confs)
        norm: Optional[PersonNorm] = None
        left_m = right_m = None
        left_raw, left_zone = ArmSummary.NO_DATA, BodyZone.UNKNOWN
        right_raw, right_zone = ArmSummary.NO_DATA, BodyZone.UNKNOWN

        if raw_norm is not None:
            norm = self._smooth_norm(raw_norm)
            refs = _torso_refs(frame, confs, norm)
            if refs is not None:
                hip_mid, torso_mid, _ = refs
                left_m = _arm_metrics(frame, confs, LS, LE, LW, LH, norm, hip_mid, torso_mid)
                right_m = _arm_metrics(frame, confs, RS, RE, RW, RH, norm, hip_mid, torso_mid)
                left_m = self._with_vel(left_m, self._prev_left)
                right_m = self._with_vel(right_m, self._prev_right)
                left_raw, left_zone = _raw_arm_summary(
                    left_m, self._prev_left, frame, confs, LS, LE, LW, LH, norm, hip_mid, torso_mid, False
                )
                right_raw, right_zone = _raw_arm_summary(
                    right_m, self._prev_right, frame, confs, RS, RE, RW, RH, norm, hip_mid, torso_mid, True
                )

        self._prev_left = left_m
        self._prev_right = right_m

        events: List[str] = []
        events.extend(self._left.update(left_raw, left_zone))
        events.extend(self._right.update(right_raw, right_zone))

        both_normal = (
            self._left.stable in (ArmSummary.CALM, ArmSummary.EXTENDED_NEAR)
            and self._right.stable in (ArmSummary.CALM, ArmSummary.EXTENDED_NEAR)
            and self._left.stable != ArmSummary.TOWARD_POCKET
            and self._right.stable != ArmSummary.TOWARD_POCKET
            and self._left.stable != ArmSummary.EXTENDED_AWAY
            and self._right.stable != ArmSummary.EXTENDED_AWAY
            and left_raw != ArmSummary.NO_DATA
            and right_raw != ArmSummary.NO_DATA
        )
        if both_normal and not self._both_calm_logged:
            events.insert(0, "Brazos cerca del cuerpo (normal)")
            self._both_calm_logged = True
        elif not both_normal:
            self._both_calm_logged = False

        return FrameMovementResult(
            left_summary=self._left.stable,
            right_summary=self._right.stable,
            left_zone=self._left.stable_zone,
            right_zone=self._right.stable_zone,
            left_metrics=left_m or ArmFrameMetrics(),
            right_metrics=right_m or ArmFrameMetrics(),
            person_norm=norm,
            events=events,
            status_line=self._status_line(norm),
        )

    def _status_line(self, norm: Optional[PersonNorm]) -> str:
        parts: List[str] = []
        any_active = any(
            t.stable in (ArmSummary.EXTENDED_AWAY, ArmSummary.TOWARD_POCKET, ArmSummary.RETRACTED)
            for t in (self._left, self._right)
        )
        both_near = (
            self._left.stable in (ArmSummary.CALM, ArmSummary.EXTENDED_NEAR)
            and self._right.stable in (ArmSummary.CALM, ArmSummary.EXTENDED_NEAR)
            and not any_active
        )
        if both_near:
            base = "Brazos cerca del cuerpo (normal)"
            if norm is not None:
                base += f"  [escala {norm.scale_ratio:.2f}]"
            return base

        for tracker in (self._right, self._left):
            st = tracker.stable
            if st == ArmSummary.TOWARD_POCKET:
                where = ZONE_LABELS_ES.get(tracker.stable_zone, "cadera")
                parts.append(f"{tracker.side_label}: → {where}")
            elif st == ArmSummary.EXTENDED_AWAY:
                parts.append(f"{tracker.side_label}: alejado del cuerpo")
            elif st == ArmSummary.RETRACTED:
                where = ZONE_LABELS_ES.get(tracker.stable_zone, "cuerpo")
                parts.append(f"{tracker.side_label}: recogido ({where})")
            elif st == ArmSummary.EXTENDED_NEAR:
                parts.append(f"{tracker.side_label}: cerca del cuerpo")
            elif st == ArmSummary.CALM:
                parts.append(f"{tracker.side_label}: tranquilo")
        line = " | ".join(parts) if parts else "—"
        if norm is not None:
            line += f"  [escala {norm.scale_ratio:.2f}]"
        return line


def build_masked_pose(
    kpts_xy_norm: np.ndarray,
    confs: np.ndarray,
    keep_indices: List[int],
    min_conf: float = MIN_KP_CONF,
) -> Tuple[np.ndarray, np.ndarray]:
    out = kpts_xy_norm[np.array(keep_indices, dtype=int)].astype(np.float64, copy=True)
    local_confs = confs[np.array(keep_indices, dtype=int)].astype(np.float64, copy=True)
    for j in range(len(out)):
        if local_confs[j] <= min_conf:
            out[j] = np.nan
    return out, local_confs
