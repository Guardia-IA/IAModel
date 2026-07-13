#!/usr/bin/env python3
"""
Heurísticas cinemáticas de robo sobre poses (8 keypoints).

Cubre dos patrones de robo:
  A) reach → conceal  (coger en estantería y ocultar)
  B) conceal-only     (objeto ya en mano; ocultación en pasillo sin reach previo)

Uso:
  cd experiments/training/campaign

  # Un clip
  python pose_robbery_heuristics.py analyze /ruta/poses.npy

  # Split val de una celda
  python pose_robbery_heuristics.py batch --cell bin_filtered_hardened --split val --run-id fp_v1

  # Unir con CSV de ensemble y barrido de umbrales
  python pose_robbery_heuristics.py pipeline \\
      --ensemble-csv artifacts/runs/fp_v1/reports/bin_filtered_hardened/val_ensemble_fp_mean_modelo_36+modelo_40_t0.68.csv \\
      --run-id fp_v1 --split val
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

CAMPAIGN_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_DIR))

from clean_theft import (  # noqa: E402
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    _arm_side_values,
    _baseline_from_edges,
    _find_local_maxima,
    _kp_visible,
    _load_pose_sequence,
    _moving_average,
    _peak_prominence,
    _robust_amplitude,
    _torso_refs,
    compute_theft_signals,
)

try:
    from campaign_paths import (
        ensure_cell_dirs,
        filter_cells,
        load_merged_campaign_config,
        training_plan_path,
    )
    from evaluate_validation import build_split_examples, load_split_uids
    from evaluate_campaign import _binary_metrics
    from class_map_utils import apply_class_map_spec, load_class_map
    from campaign_paths import class_map_path
except ImportError as exc:
    raise SystemExit(f"Import error: {exc}") from exc


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

@dataclass
class HeuristicFeatures:
    reach_peak: float
    conceal_peak: float
    low_pick_peak: float
    combined_peak: float
    reach_then_conceal: float
    conceal_only: float
    front_pocket: float
    back_pocket: float
    torso_proximity_frac: float
    retraction: float
    pose_quality: float
    conceal_sustained_frac: float
    reach_peak_frame: int
    conceal_peak_frame: int
    pattern: str
    s_kin: float
    robbery_like: bool


def _norm_peak(signal: np.ndarray, peak_val: float) -> float:
    amp = _robust_amplitude(signal)
    if amp <= 1e-9:
        return 0.0
    baseline = _baseline_from_edges(signal)
    return float(np.clip((peak_val - baseline) / amp, 0.0, 1.5))


def _pocket_scores(poses: np.ndarray, smooth_window: int = 5) -> Tuple[float, float]:
    """front_pocket, back_pocket en [0, ~1.5]."""
    t_total = len(poses)
    front_vals: List[float] = []
    back_vals: List[float] = []

    for t in range(t_total):
        refs = _torso_refs(poses[t])
        if refs is None:
            continue
        torso_h, hip_mid, torso_mid = refs
        if torso_h <= 1e-6:
            continue
        shoulder_mid_x = float(torso_mid[0])

        for wrist_idx, hip_idx, side_sign in (
            (LEFT_WRIST, LEFT_HIP, -1.0),
            (RIGHT_WRIST, RIGHT_HIP, 1.0),
        ):
            if not _kp_visible(poses[t][wrist_idx]) or not _kp_visible(poses[t][hip_idx]):
                continue
            wrist = poses[t][wrist_idx]
            hip = poses[t][hip_idx]
            dist = float(np.linalg.norm(wrist[:2] - hip[:2])) / torso_h
            if dist > 0.55:
                continue
            below = max(0.0, float(wrist[1] - hip[1])) / torso_h
            front_score = (1.0 / (dist + 0.08)) * (0.5 + 0.5 * min(below / 0.25, 1.0))
            front_vals.append(front_score)

            behind = (wrist[0] - shoulder_mid_x) * side_sign
            if behind > 0:
                back_vals.append((1.0 / (dist + 0.08)) * (0.6 + 0.4 * min(behind / 0.15, 1.0)))

    def _agg(vals: List[float]) -> float:
        if not vals:
            return 0.0
        arr = np.asarray(vals, dtype=np.float64)
        return float(np.percentile(arr, 92))

    return _agg(front_vals), _agg(back_vals)


def _torso_proximity_frac(poses: np.ndarray, *, ratio: float = 0.38) -> float:
    close = 0
    valid = 0
    for t in range(len(poses)):
        refs = _torso_refs(poses[t])
        if refs is None:
            continue
        torso_h, hip_mid, torso_mid = refs
        frame_close = False
        for wrist_idx in (LEFT_WRIST, RIGHT_WRIST):
            if not _kp_visible(poses[t][wrist_idx]):
                continue
            valid += 1
            d = float(np.linalg.norm(poses[t][wrist_idx][:2] - torso_mid[:2])) / torso_h
            if d <= ratio:
                frame_close = True
        if frame_close:
            close += 1
    if valid == 0:
        return 0.0
    return close / max(len(poses), 1)


def _conceal_sustained_frac(smoothed_conceal: np.ndarray, *, margin: float = 0.18) -> float:
    baseline = _baseline_from_edges(smoothed_conceal)
    amp = _robust_amplitude(smoothed_conceal)
    thr = baseline + margin * amp
    if amp <= 1e-9:
        return 0.0
    return float(np.mean(smoothed_conceal >= thr))


def _reach_then_conceal_score(
    smoothed_reach: np.ndarray,
    smoothed_conceal: np.ndarray,
    *,
    max_gap: int = 90,
    min_reach_prom: float = 0.08,
) -> Tuple[float, int, int]:
    """Pico reach seguido de pico conceal (orden temporal)."""
    reach_amp = _robust_amplitude(smoothed_reach)
    conceal_amp = _robust_amplitude(smoothed_conceal)
    best = 0.0
    r_frame, c_frame = -1, -1

    reach_peaks = [
        p
        for p in _find_local_maxima(smoothed_reach)
        if _peak_prominence(smoothed_reach, p) >= max(reach_amp * min_reach_prom, 0.03)
    ]
    conceal_peaks = [
        p
        for p in _find_local_maxima(smoothed_conceal)
        if _peak_prominence(smoothed_conceal, p) >= max(conceal_amp * 0.10, 0.5)
    ]

    for rp in reach_peaks:
        for cp in conceal_peaks:
            if cp < rp:
                continue
            if cp - rp > max_gap:
                continue
            r_prom = _peak_prominence(smoothed_reach, rp) / max(reach_amp, 1e-6)
            c_prom = _peak_prominence(smoothed_conceal, cp) / max(conceal_amp, 1e-6)
            gap_pen = 1.0 - min((cp - rp) / max(max_gap, 1), 1.0) * 0.35
            score = min(r_prom, 1.2) * min(c_prom, 1.2) * gap_pen
            if score > best:
                best = score
                r_frame, c_frame = rp, cp
    return best, r_frame, c_frame


def _conceal_only_score(
    smoothed_conceal: np.ndarray,
    smoothed_reach: np.ndarray,
    *,
    torso_frac: float,
    sustained_frac: float,
    front_pocket: float,
    back_pocket: float,
) -> Tuple[float, int]:
    """
    Ocultación sin reach previo fuerte (robo en pasillo con objeto ya en mano).
    """
    conceal_amp = _robust_amplitude(smoothed_conceal)
    reach_amp = _robust_amplitude(smoothed_reach)
    if conceal_amp <= 1e-9:
        return 0.0, -1

    conceal_peaks = [
        p
        for p in _find_local_maxima(smoothed_conceal)
        if _peak_prominence(smoothed_conceal, p) >= max(conceal_amp * 0.12, 0.5)
    ]
    if not conceal_peaks:
        return 0.0, -1

    best = 0.0
    best_frame = -1
    reach_baseline = _baseline_from_edges(smoothed_reach)

    for cp in conceal_peaks:
        c_prom = _peak_prominence(smoothed_conceal, cp) / max(conceal_amp, 1e-6)
        local_reach = float(np.max(smoothed_reach[max(0, cp - 45) : cp + 1]))
        reach_not_dominant = 1.0
        if reach_amp > 1e-6:
            reach_excess = max(0.0, (local_reach - reach_baseline) / reach_amp)
            if reach_excess > 0.55:
                reach_not_dominant = max(0.25, 1.0 - (reach_excess - 0.55) * 1.5)

        pocket_boost = 1.0 + 0.15 * min(front_pocket + back_pocket, 2.0)
        sustain_boost = 1.0 + 0.35 * min(sustained_frac / 0.25, 1.0)
        torso_boost = 1.0 + 0.25 * min(torso_frac / 0.20, 1.0)

        score = (
            min(c_prom, 1.3)
            * reach_not_dominant
            * pocket_boost
            * sustain_boost
            * torso_boost
        )
        if score > best:
            best = score
            best_frame = cp

    return best, best_frame


def _retraction_score(smoothed_reach: np.ndarray, reach_peak_frame: int) -> float:
    if reach_peak_frame < 0 or reach_peak_frame >= len(smoothed_reach) - 3:
        return 0.0
    amp = _robust_amplitude(smoothed_reach)
    if amp <= 1e-9:
        return 0.0
    peak_val = float(smoothed_reach[reach_peak_frame])
    tail = smoothed_reach[reach_peak_frame : min(len(smoothed_reach), reach_peak_frame + 35)]
    if len(tail) < 4:
        return 0.0
    drop = peak_val - float(np.min(tail))
    return float(np.clip(drop / amp, 0.0, 1.2))


def _pose_quality(poses: np.ndarray) -> float:
    if len(poses) == 0:
        return 0.0
    good = 0
    for frame in poses:
        if sum(_kp_visible(frame[i]) for i in range(8)) >= 6:
            good += 1
    return good / len(poses)


def analyze_poses(
    poses: np.ndarray,
    *,
    smooth_window: int = 7,
    s_kin_threshold: float = 0.42,
    min_pose_quality: float = 0.55,
) -> HeuristicFeatures:
    signals = compute_theft_signals(poses)
    smoothed = {
        "reach": _moving_average(signals.reach, smooth_window),
        "conceal": _moving_average(signals.conceal, smooth_window),
        "low_pick": _moving_average(signals.low_pick, smooth_window),
    }

    reach_peak_val = float(np.max(smoothed["reach"]))
    conceal_peak_val = float(np.max(smoothed["conceal"]))
    low_pick_peak_val = float(np.max(smoothed["low_pick"]))
    combined_peak_val = float(
        np.max(
            np.maximum(
                np.maximum(smoothed["reach"], smoothed["conceal"]),
                smoothed["low_pick"],
            )
        )
    )

    reach_peak_frame = int(np.argmax(smoothed["reach"]))
    conceal_peak_frame = int(np.argmax(smoothed["conceal"]))

    reach_then_conceal, rtc_r, rtc_c = _reach_then_conceal_score(
        smoothed["reach"], smoothed["conceal"]
    )
    torso_frac = _torso_proximity_frac(poses)
    sustained_frac = _conceal_sustained_frac(smoothed["conceal"])
    front_pocket, back_pocket = _pocket_scores(poses, smooth_window=smooth_window)
    conceal_only, co_frame = _conceal_only_score(
        smoothed["conceal"],
        smoothed["reach"],
        torso_frac=torso_frac,
        sustained_frac=sustained_frac,
        front_pocket=front_pocket,
        back_pocket=back_pocket,
    )
    retraction = _retraction_score(smoothed["reach"], rtc_r if rtc_r >= 0 else reach_peak_frame)
    pose_quality = _pose_quality(poses)

    n_reach = _norm_peak(smoothed["reach"], reach_peak_val)
    n_conceal = _norm_peak(smoothed["conceal"], conceal_peak_val)
    n_low = _norm_peak(smoothed["low_pick"], low_pick_peak_val)

    # Score compuesto: OR lógico vía max de patrones
    pattern_scores = {
        "reach_conceal": float(np.clip(reach_then_conceal, 0.0, 1.5)),
        "conceal_only": float(np.clip(conceal_only / 1.8, 0.0, 1.5)),
        "reach_retract": float(np.clip(0.55 * n_reach + 0.25 * retraction + 0.20 * n_conceal, 0.0, 1.5)),
        "low_pick": float(np.clip(n_low * 0.85, 0.0, 1.2)),
    }
    pattern = max(pattern_scores, key=pattern_scores.get)
    s_kin = float(
        max(
            pattern_scores["reach_conceal"],
            pattern_scores["conceal_only"],
            0.65 * n_reach + 0.35 * n_conceal,
            pattern_scores["low_pick"],
        )
    )
    # Refuerzo por proximidad sostenida / bolsillo
    if torso_frac >= 0.15 or sustained_frac >= 0.12:
        s_kin = min(1.5, s_kin + 0.12 * min(torso_frac / 0.25, 1.0) + 0.10 * min(sustained_frac / 0.30, 1.0))

    robbery_like = pose_quality >= min_pose_quality and s_kin >= s_kin_threshold

    return HeuristicFeatures(
        reach_peak=round(reach_peak_val, 4),
        conceal_peak=round(conceal_peak_val, 4),
        low_pick_peak=round(low_pick_peak_val, 4),
        combined_peak=round(combined_peak_val, 4),
        reach_then_conceal=round(reach_then_conceal, 4),
        conceal_only=round(conceal_only, 4),
        front_pocket=round(front_pocket, 4),
        back_pocket=round(back_pocket, 4),
        torso_proximity_frac=round(torso_frac, 4),
        retraction=round(retraction, 4),
        pose_quality=round(pose_quality, 4),
        conceal_sustained_frac=round(sustained_frac, 4),
        reach_peak_frame=reach_peak_frame,
        conceal_peak_frame=conceal_peak_frame if conceal_peak_frame >= 0 else co_frame,
        pattern=pattern,
        s_kin=round(s_kin, 4),
        robbery_like=robbery_like,
    )


def analyze_pose_path(
    pose_path: Path,
    *,
    user_index: int = 0,
    **kwargs: Any,
) -> HeuristicFeatures:
    poses, _ = _load_pose_sequence(str(pose_path), user_index=user_index)
    return analyze_poses(poses, **kwargs)


# ---------------------------------------------------------------------------
# Pipeline join + sweep
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def apply_pipeline_row(
    row: Dict[str, str],
    heur: HeuristicFeatures,
    *,
    t_stage1: float,
    rule: str,
    t_verifier: float,
    t_kin: float,
    min_conceal_sustain: float,
    require_kin: bool,
    p_verifier_key: str = "p_verifier",
) -> Dict[str, Any]:
    p_mean = _float(row, "p_mean", _float(row, "prob_pos"))
    p_v06 = _float(row, "p_modelo_06", p_mean)
    p_v14 = _float(row, "p_modelo_14", p_mean)
    p_verifier = _float(row, p_verifier_key, 0.0)

    if rule == "and":
        p_a = _float(row, "p_modelo_36", p_mean)
        p_b = _float(row, "p_modelo_40", p_mean)
        stage1 = (p_a >= t_stage1) and (p_b >= t_stage1)
    elif rule == "cascade":
        stage1 = (p_v06 >= 0.4) and (_float(row, "p_modelo_40", p_mean) >= 0.55)
    else:
        stage1 = p_mean >= t_stage1

    # Etapa 2: verificador — alta P(confusable) descarta
    confusable = p_verifier >= t_verifier if p_verifier_key in row and row.get(p_verifier_key) else False
    stage2_ok = not confusable if p_verifier > 0.0 or p_verifier_key in row else True

    # Regla temporal cinemática (no duración compra vs robo)
    temporal_ok = (
        heur.conceal_sustained_frac >= min_conceal_sustain
        or heur.reach_then_conceal >= 0.25
        or heur.conceal_only >= 0.35
        or (
            heur.pattern in ("reach_conceal", "conceal_only")
            and heur.s_kin >= t_kin * 0.85
        )
    )

    kin_ok = heur.s_kin >= t_kin if require_kin else True
    final_alarm = bool(stage1 and stage2_ok and kin_ok and temporal_ok)

    yt = int(float(row.get("true_label", row.get("yt", 0)) or 0))
    return {
        **{k: row.get(k, "") for k in ("uid", "clip_name", "clip_video_path", "clip_path", "folder_category", "pose_path")},
        "true_label": yt,
        "p_mean": round(p_mean, 4),
        "p_verifier": round(p_verifier, 4),
        "stage1": int(stage1),
        "stage2_ok": int(stage2_ok),
        "kin_ok": int(kin_ok),
        "temporal_ok": int(temporal_ok),
        "final_alarm": int(final_alarm),
        "s_kin": heur.s_kin,
        "pattern": heur.pattern,
        "conceal_only": heur.conceal_only,
        "reach_then_conceal": heur.reach_then_conceal,
        "conceal_sustained_frac": heur.conceal_sustained_frac,
        "robbery_like_heur": int(heur.robbery_like),
    }


def run_pipeline_sweep(
    ensemble_csv: Path,
    *,
    out_dir: Path,
    t_stage1_grid: Sequence[float],
    t_kin_grid: Sequence[float],
    t_verifier: float = 0.55,
    rule: str = "mean",
    min_conceal_sustain: float = 0.08,
    require_kin: bool = True,
    s_kin_threshold: float = 0.42,
) -> Dict[str, Any]:
    rows_in = _read_csv(ensemble_csv)
    if not rows_in:
        raise SystemExit(f"CSV vacío: {ensemble_csv}")

    enriched: List[Dict[str, Any]] = []
    heur_cache: Dict[str, HeuristicFeatures] = {}

    for row in rows_in:
        pose = row.get("pose_path") or row.get("uid_absolute") or ""
        if not pose or not Path(pose).is_file():
            enriched.append({**row, "s_kin": 0.0, "pattern": "missing_pose"})
            continue
        if pose not in heur_cache:
            heur_cache[pose] = analyze_pose_path(Path(pose), s_kin_threshold=s_kin_threshold)
        heur = heur_cache[pose]
        enriched.append({**row, **asdict(heur)})

    sweep_rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for t1 in t_stage1_grid:
        for tk in t_kin_grid:
            preds: List[int] = []
            y_true: List[int] = []
            detail: List[Dict[str, Any]] = []
            for row in rows_in:
                pose = row.get("pose_path") or row.get("uid_absolute") or ""
                heur = heur_cache.get(pose) or HeuristicFeatures(
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, "missing", 0.0, False
                )
                out = apply_pipeline_row(
                    row,
                    heur,
                    t_stage1=t1,
                    rule=rule,
                    t_verifier=t_verifier,
                    t_kin=tk,
                    min_conceal_sustain=min_conceal_sustain,
                    require_kin=require_kin,
                )
                detail.append(out)
                y_true.append(int(out["true_label"]))
                preds.append(int(out["final_alarm"]))

            m = _binary_metrics(np.array(y_true), np.array(preds))
            sweep_row = {
                "rule": rule,
                "t_stage1": t1,
                "t_kin": tk,
                "t_verifier": t_verifier,
                "min_conceal_sustain": min_conceal_sustain,
                "require_kin": require_kin,
                **m,
            }
            sweep_rows.append(sweep_row)
            if best is None or (
                m["fp"] < best["fp"]
                or (m["fp"] == best["fp"] and m["f1_pct"] > best["f1_pct"])
            ):
                best = {**sweep_row, "detail": detail}

    out_dir.mkdir(parents=True, exist_ok=True)
    heur_csv = out_dir / "heuristics_features.csv"
    _write_csv(
        heur_csv,
        enriched,
        fieldnames=list(enriched[0].keys()) if enriched else ["uid"],
    )
    sweep_csv = out_dir / "pipeline_sweep.csv"
    _write_csv(
        sweep_csv,
        sweep_rows,
        fieldnames=[
            "rule", "t_stage1", "t_kin", "t_verifier", "min_conceal_sustain", "require_kin",
            "tp", "fp", "fn", "tn", "f1_pct", "recall_pct", "fp_rate_pct", "support_pos", "support_neg",
        ],
    )
    if best:
        detail_csv = out_dir / "pipeline_best_detail.csv"
        _write_csv(detail_csv, best["detail"], fieldnames=list(best["detail"][0].keys()))
        summary = {k: v for k, v in best.items() if k != "detail"}
        summary_path = out_dir / "pipeline_best_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")

    return {"heuristics_csv": str(heur_csv), "sweep_csv": str(sweep_csv), "best": best}


def run_batch_cell(
    cell_id: str,
    split: str,
    run_id: Optional[str],
    out_csv: Path,
    *,
    s_kin_threshold: float = 0.42,
) -> Path:
    config = load_merged_campaign_config()
    cells = filter_cells(config, [cell_id])
    if not cells:
        raise SystemExit(f"Celda desconocida: {cell_id}")
    cell = cells[0]
    plan_path = training_plan_path(cell_id, run_id=run_id)
    split_uids, split_meta = load_split_uids(split_name=split, training_plan_path=plan_path)
    split_meta["split_name"] = split
    examples, _pool = build_split_examples(
        split_uids=split_uids,
        split_meta=split_meta,
        pose_source=cell["pose_source"],
        single_user_only=bool(config.get("single_user_only", True)),
        task=cell["task"],
    )
    cmap = load_class_map(class_map_path(cell["class_map_id"]))
    examples = apply_class_map_spec(examples, cmap)

    from train_model_operations import _example_folder_category

    robbery = int(config.get("robbery_class", 6))
    rows: List[Dict[str, Any]] = []
    for ex in examples:
        pose_path = Path(ex.pose_path)
        if not pose_path.is_file():
            continue
        heur = analyze_pose_path(pose_path, s_kin_threshold=s_kin_threshold)
        cat = _example_folder_category(ex)
        yt = 1 if cat == robbery else 0
        rows.append(
            {
                "uid": str(pose_path),
                "pose_path": str(pose_path),
                "folder_category": cat,
                "true_label": yt,
                **asdict(heur),
            }
        )

    _write_csv(out_csv, rows, fieldnames=list(rows[0].keys()) if rows else ["uid"])
    return out_csv


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Heurísticas cinemáticas de robo sobre poses")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_an = sub.add_parser("analyze", help="Analiza un .npy de poses")
    p_an.add_argument("pose_path", type=Path)
    p_an.add_argument("--s-kin-threshold", type=float, default=0.42)

    p_batch = sub.add_parser("batch", help="Batch sobre split val/test de una celda")
    p_batch.add_argument("--cell", default="bin_filtered_hardened")
    p_batch.add_argument("--split", choices=["val", "test"], default="val")
    p_batch.add_argument("--run-id", default=None)
    p_batch.add_argument("--out", type=Path, default=None)
    p_batch.add_argument("--s-kin-threshold", type=float, default=0.42)

    p_pipe = sub.add_parser("pipeline", help="Une CSV ensemble + heurísticas + barrido")
    p_pipe.add_argument("--ensemble-csv", type=Path, required=True)
    p_pipe.add_argument("--out-dir", type=Path, default=None)
    p_pipe.add_argument("--rule", choices=["mean", "and", "cascade"], default="mean")
    p_pipe.add_argument("--t-verifier", type=float, default=0.55)
    p_pipe.add_argument("--min-conceal-sustain", type=float, default=0.08)
    p_pipe.add_argument("--no-require-kin", action="store_true")

    args = ap.parse_args()

    if args.cmd == "analyze":
        heur = analyze_pose_path(args.pose_path, s_kin_threshold=args.s_kin_threshold)
        print(json.dumps(asdict(heur), indent=2, ensure_ascii=False))
        return 0

    if args.cmd == "batch":
        arts = ensure_cell_dirs(args.cell, run_id=args.run_id)
        out = args.out or (arts["reports_dir"] / f"{args.split}_heuristics_features.csv")
        run_batch_cell(args.cell, args.split, args.run_id, out, s_kin_threshold=args.s_kin_threshold)
        print(f"Heurísticas → {out}")
        return 0

    if args.cmd == "pipeline":
        out_dir = args.out_dir or args.ensemble_csv.parent / "fp_pipeline"
        t1_grid = [0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78]
        tk_grid = [0.30, 0.35, 0.40, 0.42, 0.45, 0.50, 0.55]
        result = run_pipeline_sweep(
            args.ensemble_csv,
            out_dir=out_dir,
            t_stage1_grid=t1_grid,
            t_kin_grid=tk_grid,
            rule=args.rule,
            t_verifier=args.t_verifier,
            min_conceal_sustain=args.min_conceal_sustain,
            require_kin=not args.no_require_kin,
        )
        best = result.get("best") or {}
        print(json.dumps({k: v for k, v in best.items() if k != "detail"}, indent=2, ensure_ascii=False))
        print(f"\nSweep → {result['sweep_csv']}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
