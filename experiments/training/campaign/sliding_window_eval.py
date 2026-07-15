#!/usr/bin/env python3
"""
Evalúa post-filtros temporales + cinemáticos sobre ventanas deslizantes de pose.

Flujo:
  1) Baseline: predicción clip completo (como evaluate_campaign).
  2) Ventanas (p.ej. 3 s, stride 1 s) → p_robo / clase por ventana.
  3) Reglas R1–R5 → alarma final (orientado a bajar FP).

Uso (modelo_12 binario, campaña yolo26m):
  cd experiments/training/campaign
  export GUADIA_DATA_RESULT_ROOT=/ruta/data_yolo26m/data_result

  # Eval completo val + barrido agresivo anti-FP
  python sliding_window_eval.py \\
      --run-id campaign_20260714_164642 \\
      --cell bin_full --model modelo_12 --split val --sweep --max-fp-target 1

  # Solo clips FP/FN del CSV de errores
  python sliding_window_eval.py \\
      --run-id campaign_20260714_164642 \\
      --cell bin_full --model modelo_12 \\
      --errors-csv artifacts/runs/.../reports/bin_full/val_errors_modelo_12.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

CAMPAIGN_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_DIR))

try:
    from campaign_paths import ensure_cell_dirs, load_campaign_config, training_plan_path
    from evaluate_validation import build_split_examples, load_split_uids
    from evaluate_campaign import _binary_metrics, load_best_ensemble_spec
    from export_fp_artifacts import example_export_paths
    from train_model_operations import (
        PoseExample,
        add_velocity,
        build_model,
        normalize_sequence,
        temporal_resize,
    )
    from pose_robbery_heuristics import analyze_poses
except ImportError as exc:
    raise SystemExit(f"Import error (¿entorno con torch?): {exc}") from exc


DEFAULT_FPS = 12.0
PURCHASE_CLASSES = frozenset({3, 4, 5})
LEAVE_CLASSES = frozenset({0})


@dataclass
class WindowPrediction:
    win_idx: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    p_robo: float
    pred_class: int
    s_kin: float
    robbery_like: bool
    conceal_peak: float
    reach_then_conceal: float


@dataclass
class TemporalPolicy:
    window_sec: float = 3.0
    stride_sec: float = 1.0
    fps: float = DEFAULT_FPS
    min_window_frames: int = 24

    p_window_threshold: float = 0.50
    full_clip_threshold: float = 0.50
    use_full_clip_baseline: bool = True
    alarm_mode: str = "filter_baseline"  # filter_baseline | windows_only

    min_consecutive_windows: int = 2
    post_purchase_veto_windows: int = 3
    purchase_classes: Tuple[int, ...] = (3, 4, 5)
    leave_classes: Tuple[int, ...] = (0,)

    require_robbery_like: bool = True
    min_s_kin: float = 0.42
    min_pose_quality: float = 0.55
    min_conceal_sustain: float = 0.0
    require_reach_then_conceal_or_conceal: bool = False

    veto_isolated_spike: bool = True
    require_multiclass_for_veto: bool = False


@dataclass
class ClipTimeline:
    uid: str
    true_label: int
    folder_category: int
    pose_path: str
    clip_path: str
    fps: float
    n_frames: int
    baseline_p_robo: float
    baseline_alarm: bool
    windows: List[WindowPrediction] = field(default_factory=list)
    final_alarm: bool = False
    filter_reason: str = ""
    kept_fp: bool = False
    lost_tp: bool = False


def _normalize_model_name(name: str) -> str:
    name = str(name).strip()
    if name.isdigit():
        return f"modelo_{int(name):02d}.pt"
    if not name.endswith(".pt"):
        return f"modelo_{name}.pt" if not name.startswith("modelo_") else f"{name}.pt"
    return name


def _read_fps(meta_json: Optional[Path], default: float = DEFAULT_FPS) -> float:
    if meta_json is None or not meta_json.is_file():
        return default
    try:
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        fps = float(meta.get("fps") or meta.get("video_fps") or default)
        return fps if fps > 0 else default
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


def _load_poses(example: Any) -> Tuple[np.ndarray, Optional[Path]]:
    paths = example_export_paths(example)
    pose_path = Path(paths["pose_path"])
    valid_mask_path = Path(paths.get("valid_mask_path") or "")
    if not pose_path.is_file():
        raise FileNotFoundError(f"Falta pose: {pose_path}")
    poses = np.load(pose_path)
    if valid_mask_path.is_file():
        vm = np.load(valid_mask_path)
        poses = poses[vm].copy()
    if np.any(np.isnan(poses)):
        poses = np.nan_to_num(poses, nan=0.0, posinf=0.0, neginf=0.0)
    return poses, valid_mask_path if valid_mask_path.is_file() else None


def _idx_for_class(label_to_idx: Dict[Any, int], class_id: int) -> Optional[int]:
    for key in (class_id, str(class_id)):
        if key in label_to_idx:
            return int(label_to_idx[key])
    return None


def _label_from_idx(label_to_idx: Dict[Any, int], idx: int) -> int:
    for k, v in label_to_idx.items():
        if int(v) == int(idx):
            try:
                return int(k)
            except (TypeError, ValueError):
                continue
    return int(idx)


class WindowModelRunner:
    def __init__(self, model_path: Path, device: Optional[torch.device] = None):
        self.model_path = model_path.resolve()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.checkpoint = ckpt
        self.label_to_idx = ckpt["label_to_idx"]
        self.seq_len = int(ckpt.get("seq_len", 64))
        cfg = ckpt.get("config", {})
        self.arch = cfg.get("arch", "tcn")
        self.input_dim = int(ckpt["input_dim"])
        self.num_classes = int(ckpt.get("num_classes", len(self.label_to_idx)))
        self.is_binary = self.num_classes == 2
        self.robbery_idx = _idx_for_class(self.label_to_idx, int(ckpt.get("positive_class", 6)))
        if self.is_binary:
            self.robbery_idx = 1

        model = build_model(self.arch, self.input_dim, self.num_classes, cfg).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        self.model = model

    @torch.no_grad()
    def predict_poses(self, poses: np.ndarray) -> Tuple[float, int, np.ndarray]:
        if len(poses) < 2:
            return 0.0, 0, np.zeros(self.num_classes, dtype=np.float64)
        proc = normalize_sequence(poses.astype(np.float32).copy())
        proc = add_velocity(proc)
        proc = temporal_resize(proc, self.seq_len)
        t, j, d = proc.shape
        x = torch.from_numpy(proc.reshape(t, j * d).astype(np.float32)).unsqueeze(0).to(self.device)
        logits = self.model(x)
        prob = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
        if self.is_binary:
            p_robo = float(prob[1])
            pred_class = 6 if p_robo >= 0.5 else 0
        else:
            argmax_idx = int(np.argmax(prob))
            pred_class = _label_from_idx(self.label_to_idx, argmax_idx)
            if self.robbery_idx is not None:
                p_robo = float(prob[self.robbery_idx])
            else:
                p_robo = float(prob[argmax_idx]) if pred_class == 6 else 0.0
        return p_robo, pred_class, prob


@dataclass
class EnsembleSpec:
    models: List[str]
    rule: str
    thresholds: List[float]
    source: str = "explicit"
    f1_pct: Optional[float] = None
    fp: Optional[int] = None
    fn: Optional[int] = None

    @property
    def threshold(self) -> float:
        return float(self.thresholds[0]) if self.thresholds else 0.5

    def file_tag(self) -> str:
        model_tag = "+".join(m.replace(".pt", "") for m in self.models)
        if self.rule == "mean":
            t = f"{self.threshold:.2f}"
            return f"ensemble_{self.rule}_{model_tag}_t{t}"
        t = "+".join(f"{x:.2f}" for x in self.thresholds)
        return f"ensemble_{self.rule}_{model_tag}_t{t}"


def _ensemble_spec_from_json(spec: Dict[str, Any], source: str = "best_low_fp") -> EnsembleSpec:
    models = [str(m) for m in spec.get("models") or []]
    rule = str(spec.get("rule") or "mean")
    thr = spec.get("threshold")
    if isinstance(thr, list):
        thresholds = [float(x) for x in thr]
    elif thr is not None:
        thresholds = [float(thr)] * max(len(models), 1)
    else:
        thrs = spec.get("thresholds")
        thresholds = [float(x) for x in thrs] if isinstance(thrs, list) else [0.5]
    if rule == "and" and len(thresholds) == 1 and len(models) > 1:
        thresholds = thresholds * len(models)
    return EnsembleSpec(
        models=models,
        rule=rule,
        thresholds=thresholds,
        source=source,
        f1_pct=float(spec["f1_pct"]) if spec.get("f1_pct") is not None else None,
        fp=int(spec["fp"]) if spec.get("fp") is not None else None,
        fn=int(spec["fn"]) if spec.get("fn") is not None else None,
    )


def _ensemble_spec_from_grid_row(row: Dict[str, str], source: str = "best_f1") -> EnsembleSpec:
    rule = str(row.get("decision_mode", "ensemble_mean")).replace("ensemble_", "")
    models = [m.strip() for m in str(row.get("models", "")).split("|") if m.strip()]
    thresholds = [float(t) for t in str(row.get("thresholds", "0.5")).split("|") if t.strip()]
    if rule == "mean" and len(thresholds) > 1:
        thresholds = [thresholds[0]]
    if rule == "and" and len(thresholds) == 1 and len(models) > 1:
        thresholds = thresholds * len(models)
    return EnsembleSpec(
        models=models,
        rule=rule,
        thresholds=thresholds,
        source=source,
        f1_pct=float(row.get("f1_pct", 0) or 0),
        fp=int(float(row.get("fp", 0) or 0)),
        fn=int(float(row.get("fn", 0) or 0)),
    )


def load_best_f1_ensemble_spec(reports_dir: Path, split: str) -> EnsembleSpec:
    grid = reports_dir / f"{split}_ensemble_grid.csv"
    if not grid.is_file():
        raise FileNotFoundError(
            f"No hay {grid.name}. Ejecuta evaluate_campaign.py antes o usa --ensemble-models explícito."
        )
    rows = list(csv.DictReader(grid.open(encoding="utf-8")))
    ens_rows = [r for r in rows if str(r.get("decision_mode", "")).startswith("ensemble_")]
    if not ens_rows:
        raise RuntimeError(f"Sin filas ensemble en {grid}")
    best = max(ens_rows, key=lambda r: float(r.get("f1_pct", 0) or 0))
    return _ensemble_spec_from_grid_row(best, source="best_f1")


def resolve_ensemble_spec(
    args: argparse.Namespace,
    reports_dir: Path,
    split: str,
) -> EnsembleSpec:
    if args.ensemble_models:
        rule = str(args.ensemble_rule or "mean")
        thrs = (
            [float(args.ensemble_threshold)] * len(args.ensemble_models)
            if rule == "and"
            else [float(args.ensemble_threshold)]
        )
        return EnsembleSpec(
            models=[_normalize_model_name(m).replace(".pt", "") for m in args.ensemble_models],
            rule=rule,
            thresholds=thrs,
            source="explicit",
        )
    source = str(args.ensemble_source or "best_f1")
    if source == "best_f1":
        return load_best_f1_ensemble_spec(reports_dir, split)
    if source in ("best_low_fp", "auto"):
        spec = load_best_ensemble_spec(reports_dir, split)
        if spec is None:
            raise FileNotFoundError(f"No hay {split}_best_ensemble.json en {reports_dir}")
        return _ensemble_spec_from_json(spec, source=source)
    raise ValueError(f"ensemble_source desconocido: {source!r}")


class WindowEnsembleRunner:
    """Ensemble binario: mean / and sobre p_robo de varios checkpoints."""

    def __init__(self, models_dir: Path, spec: EnsembleSpec, device: Optional[torch.device] = None):
        self.spec = spec
        self.tag = spec.file_tag()
        self.models_dir = models_dir
        self.runners: List[WindowModelRunner] = []
        for name in spec.models:
            path = models_dir / _normalize_model_name(name)
            if not path.is_file():
                raise FileNotFoundError(f"No existe modelo ensemble: {path}")
            self.runners.append(WindowModelRunner(path, device=device))
        if not self.runners:
            raise ValueError("Ensemble sin modelos")
        self.is_binary = all(r.is_binary for r in self.runners)

    def _combine(self, p_list: Sequence[float]) -> Tuple[float, bool]:
        rule = self.spec.rule.lower()
        thrs = self.spec.thresholds
        if rule == "mean":
            p = float(np.mean(p_list))
            thr = thrs[0] if thrs else 0.5
            return p, p >= thr
        if rule == "and":
            if len(thrs) < len(p_list):
                thrs = thrs + [thrs[-1]] * (len(p_list) - len(thrs))
            ok = all(p >= t for p, t in zip(p_list, thrs))
            p = float(np.mean(p_list))
            return p, ok
        raise ValueError(f"Regla ensemble no soportada: {rule!r} (mean|and)")

    @torch.no_grad()
    def predict_poses(self, poses: np.ndarray) -> Tuple[float, int, np.ndarray]:
        if len(poses) < 2:
            return 0.0, 0, np.zeros(2, dtype=np.float64)
        p_list: List[float] = []
        pred_classes: List[int] = []
        for runner in self.runners:
            p_robo, pred_class, _ = runner.predict_poses(poses)
            p_list.append(p_robo)
            pred_classes.append(pred_class)
        p_robo, alarm = self._combine(p_list)
        pred_class = 6 if alarm else int(pred_classes[0] if pred_classes else 0)
        return p_robo, pred_class, np.array(p_list, dtype=np.float64)


def make_predictor(
    args: argparse.Namespace,
    arts: Dict[str, Path],
) -> Tuple[Any, TemporalPolicy]:
    policy = TemporalPolicy(
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        fps=args.fps,
        p_window_threshold=args.p_window_threshold,
        full_clip_threshold=args.full_clip_threshold,
        min_consecutive_windows=args.min_consecutive_windows,
        min_s_kin=args.min_s_kin,
        require_robbery_like=not args.no_require_kin,
        require_reach_then_conceal_or_conceal=args.require_conceal,
        post_purchase_veto_windows=args.post_purchase_veto_windows,
        alarm_mode=args.alarm_mode,
        use_full_clip_baseline=not args.windows_only,
        veto_isolated_spike=not args.allow_isolated_spike,
    )

    predictor = str(args.predictor or "single").lower()
    if predictor == "single":
        model_path = arts["models_dir"] / _normalize_model_name(args.model)
        if not model_path.is_file():
            raise FileNotFoundError(f"No existe modelo: {model_path}")
        runner: Any = WindowModelRunner(model_path)
        runner.tag = model_path.stem  # type: ignore[attr-defined]
        print(f"  Checkpoint: {model_path} (binario={runner.is_binary})", flush=True)
        return runner, policy

    spec = resolve_ensemble_spec(args, arts["reports_dir"], args.split)
    runner = WindowEnsembleRunner(arts["models_dir"], spec)
    for r in runner.runners:
        print(f"  Checkpoint ensemble: {r.model_path}", flush=True)
    thr = spec.threshold
    policy.full_clip_threshold = thr
    if args.p_window_threshold == 0.50:
        policy.p_window_threshold = thr
    print(
        f"  Ensemble ({spec.source}): {'|'.join(spec.models)} {spec.rule} @ {spec.thresholds} "
        f"| val F1={spec.f1_pct}% FP={spec.fp} FN={spec.fn}",
        flush=True,
    )
    return runner, policy


def iter_windows(
    n_frames: int,
    *,
    window_frames: int,
    stride_frames: int,
    min_window_frames: int,
) -> Iterable[Tuple[int, int, int]]:
    if n_frames < min_window_frames:
        yield 0, 0, n_frames
        return
    if n_frames <= window_frames:
        yield 0, 0, n_frames
        return
    start = 0
    win_idx = 0
    while start + min_window_frames <= n_frames:
        end = min(start + window_frames, n_frames)
        if end - start >= min_window_frames:
            yield win_idx, start, end
            win_idx += 1
        if end >= n_frames:
            break
        start += stride_frames


def build_timeline(
    example: Any,
    runner: Any,
    policy: TemporalPolicy,
    *,
    baseline_p: Optional[float] = None,
) -> ClipTimeline:
    paths = example_export_paths(example)
    uid = paths["uid"]
    poses, _ = _load_poses(example)
    n_frames = len(poses)
    meta_path = Path(paths["meta_json_path"]) if paths.get("meta_json_path") else None
    fps = _read_fps(meta_path, policy.fps)

    p_full, cls_full, _ = runner.predict_poses(poses)
    baseline_p = float(baseline_p if baseline_p is not None else p_full)
    if isinstance(runner, WindowEnsembleRunner):
        baseline_alarm = int(cls_full) == 6
    else:
        baseline_alarm = baseline_p >= policy.full_clip_threshold

    window_frames = max(int(round(policy.window_sec * fps)), policy.min_window_frames)
    stride_frames = max(1, int(round(policy.stride_sec * fps)))

    windows: List[WindowPrediction] = []
    for win_idx, start, end in iter_windows(
        n_frames,
        window_frames=window_frames,
        stride_frames=stride_frames,
        min_window_frames=policy.min_window_frames,
    ):
        slice_poses = poses[start:end]
        p_robo, pred_class, _ = runner.predict_poses(slice_poses)
        heur = analyze_poses(
            slice_poses,
            s_kin_threshold=policy.min_s_kin,
            min_pose_quality=policy.min_pose_quality,
        )
        windows.append(
            WindowPrediction(
                win_idx=win_idx,
                start_frame=start,
                end_frame=end,
                start_sec=round(start / fps, 3),
                end_sec=round(end / fps, 3),
                p_robo=round(p_robo, 6),
                pred_class=int(pred_class),
                s_kin=heur.s_kin,
                robbery_like=bool(heur.robbery_like),
                conceal_peak=heur.conceal_peak,
                reach_then_conceal=heur.reach_then_conceal,
            )
        )

    from train_model_operations import _example_folder_category

    folder_cat = int(_example_folder_category(example))
    yt = 1 if folder_cat == 6 else 0

    tl = ClipTimeline(
        uid=uid,
        true_label=yt,
        folder_category=folder_cat,
        pose_path=str(paths["pose_path"]),
        clip_path=str(paths["clip_video_path"] or paths["clip_dir"]),
        fps=fps,
        n_frames=n_frames,
        baseline_p_robo=round(baseline_p, 6),
        baseline_alarm=baseline_alarm,
        windows=windows,
    )
    final, reason = apply_temporal_policy(tl, policy, runner.is_binary)
    tl.final_alarm = final
    tl.filter_reason = reason
    tl.kept_fp = baseline_alarm and tl.true_label == 0 and final
    tl.lost_tp = baseline_alarm and tl.true_label == 1 and not final
    return tl


def _window_hot(w: WindowPrediction, policy: TemporalPolicy) -> bool:
    if w.p_robo < policy.p_window_threshold:
        return False
    if policy.require_robbery_like and not w.robbery_like:
        return False
    if w.s_kin < policy.min_s_kin:
        return False
    if policy.require_reach_then_conceal_or_conceal:
        if w.reach_then_conceal < 0.25 and w.conceal_peak < 0.15:
            return False
    return True


def _has_consecutive_hot(windows: Sequence[WindowPrediction], policy: TemporalPolicy) -> bool:
    run = 0
    for w in windows:
        if _window_hot(w, policy):
            run += 1
            if run >= policy.min_consecutive_windows:
                return True
        else:
            run = 0
    return policy.min_consecutive_windows <= 1 and any(_window_hot(w, policy) for w in windows)


def _post_purchase_veto(windows: Sequence[WindowPrediction], policy: TemporalPolicy) -> bool:
    """True si hay veto (no robo): pico de robo seguido de compra."""
    if not windows or policy.post_purchase_veto_windows <= 0:
        return False
    purchase = set(policy.purchase_classes)
    for i, w in enumerate(windows):
        if not _window_hot(w, policy):
            continue
        tail = windows[i + 1 : i + 1 + policy.post_purchase_veto_windows]
        if not tail:
            continue
        if any(t.pred_class in purchase for t in tail):
            return True
    return False


def _isolated_spike(windows: Sequence[WindowPrediction], policy: TemporalPolicy) -> bool:
    if not policy.veto_isolated_spike or policy.min_consecutive_windows > 1:
        return False
    hot = [_window_hot(w, policy) for w in windows]
    if not any(hot):
        return False
    for i, h in enumerate(hot):
        if not h:
            continue
        prev_h = hot[i - 1] if i > 0 else False
        next_h = hot[i + 1] if i + 1 < len(hot) else False
        if not prev_h and not next_h:
            return True
    return False


def apply_temporal_policy(
    tl: ClipTimeline,
    policy: TemporalPolicy,
    is_binary: bool,
) -> Tuple[bool, str]:
    windows = tl.windows
    max_win_p = max((w.p_robo for w in windows), default=0.0)

    if policy.alarm_mode == "windows_only":
        candidate = _has_consecutive_hot(windows, policy) or (
            max_win_p >= policy.p_window_threshold and policy.min_consecutive_windows <= 1
        )
        if not candidate:
            return False, "no_hot_window"
    else:
        if policy.use_full_clip_baseline:
            candidate = tl.baseline_alarm
        else:
            candidate = max_win_p >= policy.p_window_threshold
        if not candidate:
            return False, "baseline_off"

    if not _has_consecutive_hot(windows, policy) and policy.min_consecutive_windows > 1:
        return False, "no_consecutive_windows"

    if policy.require_robbery_like:
        hot_kin = [w for w in windows if _window_hot(w, policy)]
        if not hot_kin:
            return False, "kinematics_fail"

    if policy.require_reach_then_conceal_or_conceal:
        if not any(w.reach_then_conceal >= 0.25 or w.conceal_peak >= 0.20 for w in windows if _window_hot(w, policy)):
            return False, "no_conceal_pattern"

    if _isolated_spike(windows, policy):
        return False, "isolated_spike"

    if not is_binary:
        if _post_purchase_veto(windows, policy):
            return False, "post_purchase_veto"

    return True, "alarm"


def evaluate_timelines(
    timelines: Sequence[ClipTimeline],
    *,
    label: str = "",
) -> Dict[str, Any]:
    y_true = np.array([t.true_label for t in timelines], dtype=np.int64)
    baseline_pred = np.array([1 if t.baseline_alarm else 0 for t in timelines], dtype=np.int64)
    final_pred = np.array([1 if t.final_alarm else 0 for t in timelines], dtype=np.int64)
    base_m = _binary_metrics(y_true, baseline_pred)
    final_m = _binary_metrics(y_true, final_pred)

    baseline_fps = [t for t in timelines if t.baseline_alarm and t.true_label == 0]
    final_fps = [t for t in timelines if t.final_alarm and t.true_label == 0]
    removed_fps = [t for t in timelines if t.baseline_alarm and t.true_label == 0 and not t.final_alarm]
    lost_tps = [t for t in timelines if t.baseline_alarm and t.true_label == 1 and not t.final_alarm]

    return {
        "label": label,
        "n_clips": len(timelines),
        "baseline": base_m,
        "filtered": final_m,
        "baseline_fp_count": len(baseline_fps),
        "filtered_fp_count": len(final_fps),
        "fp_removed": len(removed_fps),
        "tp_lost": len(lost_tps),
        "fp_removed_uids": [t.uid for t in removed_fps],
        "tp_lost_uids": [t.uid for t in lost_tps],
        "remaining_fp_uids": [t.uid for t in final_fps],
    }


def _policy_grid(aggressive: bool) -> List[TemporalPolicy]:
    policies: List[TemporalPolicy] = []
    if aggressive:
        p_thr = [0.35, 0.45, 0.50, 0.55, 0.65]
        full_thr = [0.40, 0.50, 0.55, 0.65]
        min_cons = [1, 2, 3]
        min_kin = [0.35, 0.42, 0.50, 0.58]
        req_kin = [True, False]
        req_conceal = [False, True]
        post_veto = [2, 3, 4]
        modes = ["filter_baseline", "windows_only"]
    else:
        p_thr = [0.45, 0.50, 0.55]
        full_thr = [0.50, 0.55]
        min_cons = [1, 2]
        min_kin = [0.38, 0.42, 0.48]
        req_kin = [True]
        req_conceal = [False]
        post_veto = [2, 3]
        modes = ["filter_baseline"]

    for mode in modes:
        for pt in p_thr:
            for ft in full_thr:
                for mc in min_cons:
                    for mk in min_kin:
                        for rk in req_kin:
                            for rc in req_conceal:
                                for pv in post_veto:
                                    policies.append(
                                        TemporalPolicy(
                                            alarm_mode=mode,
                                            p_window_threshold=pt,
                                            full_clip_threshold=ft,
                                            min_consecutive_windows=mc,
                                            min_s_kin=mk,
                                            require_robbery_like=rk,
                                            require_reach_then_conceal_or_conceal=rc,
                                            post_purchase_veto_windows=pv,
                                        )
                                    )
    return policies


def sweep_policies(
    timelines_baseline: Sequence[ClipTimeline],
    *,
    is_binary: bool,
    max_fp_target: int = 1,
    min_recall_pct: float = 0.0,
) -> List[Dict[str, Any]]:
    """Re-aplica políticas sobre timelines ya calculados (sin re-inferencia)."""
    results: List[Dict[str, Any]] = []
    for policy in _policy_grid(aggressive=True):
        filtered: List[ClipTimeline] = []
        for tl in timelines_baseline:
            final, reason = apply_temporal_policy(tl, policy, is_binary)
            copy = ClipTimeline(**{**tl.__dict__, "windows": list(tl.windows)})
            copy.final_alarm = final
            copy.filter_reason = reason
            filtered.append(copy)
        ev = evaluate_timelines(filtered, label="sweep")
        fm = ev["filtered"]
        if fm["recall_pct"] < min_recall_pct:
            continue
        row = {
            **{k: v for k, v in asdict(policy).items()},
            "fp": fm["fp"],
            "fn": fm["fn"],
            "tp": fm["tp"],
            "f1_pct": fm["f1_pct"],
            "recall_pct": fm["recall_pct"],
            "fp_rate_pct": fm["fp_rate_pct"],
            "fp_removed": ev["fp_removed"],
            "tp_lost": ev["tp_lost"],
            "meets_fp_target": fm["fp"] <= max_fp_target,
        }
        results.append(row)
    results.sort(key=lambda r: (r["fp"], -r["recall_pct"], -r["f1_pct"]))
    return results


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _load_errors_uids(csv_path: Path) -> Optional[set]:
    if not csv_path.is_file():
        return None
    uids: set = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = row.get("uid") or row.get("uid_absolute")
            if uid:
                uids.add(str(uid))
    return uids if uids else None


def _execute_eval(
    args: argparse.Namespace,
    *,
    runner: Any,
    policy: TemporalPolicy,
    tag: str,
    examples: Sequence[Any],
    pool_info: Dict[str, Any],
    reports_dir: Path,
    cell_id: str,
) -> Dict[str, Any]:
    is_binary = bool(getattr(runner, "is_binary", True))
    kind = "ensemble" if isinstance(runner, WindowEnsembleRunner) else "single"
    print(
        f"  Predictor={kind} tag={tag} ({'binario' if is_binary else 'multiclase'}) "
        f"| ventana={args.window_sec}s | clips={len(examples)}",
        flush=True,
    )

    timelines: List[ClipTimeline] = []
    for i, ex in enumerate(examples):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{tag}] Timeline {i + 1}/{len(examples)}...", flush=True)
        timelines.append(build_timeline(ex, runner, policy))

    summary = evaluate_timelines(timelines, label=tag)
    summary["policy"] = asdict(policy)
    summary["pool_info"] = pool_info
    summary["predictor_tag"] = tag
    summary["predictor_kind"] = kind
    if isinstance(runner, WindowEnsembleRunner):
        summary["ensemble"] = asdict(runner.spec)
    else:
        summary["model"] = tag
    summary["cell_id"] = cell_id
    summary["split"] = args.split

    out_json = reports_dir / f"{args.split}_sliding_window_{tag}_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    detail_rows: List[Dict[str, Any]] = []
    for tl in timelines:
        detail_rows.append(
            {
                "uid": tl.uid,
                "true_label": tl.true_label,
                "folder_category": tl.folder_category,
                "baseline_p_robo": tl.baseline_p_robo,
                "baseline_alarm": int(tl.baseline_alarm),
                "final_alarm": int(tl.final_alarm),
                "filter_reason": tl.filter_reason,
                "n_windows": len(tl.windows),
                "max_window_p": max((w.p_robo for w in tl.windows), default=0.0),
                "clip_path": tl.clip_path,
                "pose_path": tl.pose_path,
            }
        )
    detail_csv = reports_dir / f"{args.split}_sliding_window_{tag}_clips.csv"
    _write_csv(
        detail_csv,
        detail_rows,
        fieldnames=list(detail_rows[0].keys()) if detail_rows else ["uid"],
    )

    fp_removed_path = reports_dir / f"{args.split}_sliding_window_{tag}_fp_removed.txt"
    remaining_fp_path = reports_dir / f"{args.split}_sliding_window_{tag}_fp_remaining.txt"
    fn_lost_path = reports_dir / f"{args.split}_sliding_window_{tag}_tp_lost.txt"

    def _write_uid_list(path: Path, uids: Sequence[str], header: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {header}\n")
            for u in uids:
                f.write(f"{u}\n")

    _write_uid_list(fp_removed_path, summary["fp_removed_uids"], "FP eliminados por filtro temporal/cinemático")
    _write_uid_list(remaining_fp_path, summary["remaining_fp_uids"], "FP que siguen tras filtro")
    _write_uid_list(fn_lost_path, summary["tp_lost_uids"], "Robos perdidos en baseline pero filtrados")

    bm = summary["baseline"]
    fm = summary["filtered"]
    print(f"\n=== {tag} — split {args.split} ===")
    print(
        f"BASELINE  TP={bm['tp']} FP={bm['fp']} FN={bm['fn']} | "
        f"F1={bm['f1_pct']:.1f}% Rec={bm['recall_pct']:.1f}% FP%={bm['fp_rate_pct']:.2f}"
    )
    print(
        f"FILTRADO  TP={fm['tp']} FP={fm['fp']} FN={fm['fn']} | "
        f"F1={fm['f1_pct']:.1f}% Rec={fm['recall_pct']:.1f}% FP%={fm['fp_rate_pct']:.2f}"
    )
    print(f"  FP eliminados: {summary['fp_removed']} | Robos perdidos: {summary['tp_lost']}")
    print(f"  Resumen: {out_json}")

    if args.sweep:
        print(f"\n=== Barrido [{tag}] ===", flush=True)
        sweep_rows = sweep_policies(
            timelines,
            is_binary=is_binary,
            max_fp_target=args.max_fp_target,
            min_recall_pct=args.min_recall_pct,
        )
        sweep_csv = reports_dir / f"{args.split}_sliding_window_{tag}_sweep.csv"
        if sweep_rows:
            fields = list(sweep_rows[0].keys())
            _write_csv(sweep_csv, sweep_rows[:500], fieldnames=fields)
            best = sweep_rows[0]
            print(
                f"  Mejor barrido: FP={best['fp']} FN={best['fn']} F1={best['f1_pct']:.1f}% "
                f"Rec={best['recall_pct']:.1f}%"
            )
            hits = [r for r in sweep_rows if r["meets_fp_target"]]
            if hits:
                h = hits[0]
                print(f"  ✓ FP≤{args.max_fp_target}: F1={h['f1_pct']:.1f}% Rec={h['recall_pct']:.1f}%")
            else:
                print(f"  ⚠ Ninguna config alcanza FP≤{args.max_fp_target}")
            print(f"  CSV: {sweep_csv}")

    summary["paths"] = {
        "summary_json": str(out_json),
        "clips_csv": str(detail_csv),
        "fp_remaining": str(remaining_fp_path),
    }
    return summary


def run_eval(args: argparse.Namespace) -> int:
    config = load_campaign_config()
    cell = next((c for c in config.get("cells", []) if c["id"] == args.cell), None)
    if cell is None:
        raise SystemExit(f"Celda desconocida: {args.cell}")

    print(
        f"  Celda={args.cell} task={cell.get('task')} pose={cell.get('pose_source')} "
        f"→ models/{args.cell}/",
        flush=True,
    )

    arts = ensure_cell_dirs(args.cell, run_id=args.run_id)
    plan_path = training_plan_path(args.cell, run_id=args.run_id)
    if not plan_path.is_file():
        raise SystemExit(f"Falta plan: {plan_path}")

    model_path = arts["models_dir"] / _normalize_model_name(args.model)
    pred_mode = str(args.predictor or "single").lower()
    if pred_mode in ("single", "both") and not model_path.is_file():
        raise SystemExit(f"No existe modelo: {model_path}")
    if pred_mode == "ensemble":
        try:
            resolve_ensemble_spec(args, arts["reports_dir"], args.split)
        except (FileNotFoundError, RuntimeError) as exc:
            raise SystemExit(str(exc)) from exc

    split_uids, split_meta = load_split_uids(split_name=args.split, training_plan_path=plan_path)
    split_meta["split_name"] = args.split
    with open(plan_path, encoding="utf-8") as f:
        plan = json.load(f)
    split_meta["split_uids_all"] = {k: [str(x) for x in v] for k, v in plan.get("split_uids", {}).items()}

    su = bool(config.get("single_user_only", True))
    examples, pool_info = build_split_examples(
        split_uids=split_uids,
        split_meta=split_meta,
        pose_source=cell["pose_source"],
        single_user_only=su,
        task=cell["task"],
    )
    if not examples:
        raise SystemExit(f"Sin ejemplos en split {args.split}")

    filter_uids = _load_errors_uids(Path(args.errors_csv)) if args.errors_csv else None
    if filter_uids:
        from train_model_operations import _example_uid

        examples = [ex for ex in examples if _example_uid(ex) in filter_uids or _example_uid(ex).split("/")[-1] in filter_uids]
        print(f"  Filtrado errors-csv: {len(examples)} clips", flush=True)

    reports_dir = arts["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    predictor_mode = str(args.predictor or "single").lower()
    jobs: List[Tuple[str, argparse.Namespace]] = []
    if predictor_mode == "both":
        single_args = argparse.Namespace(**{**vars(args), "predictor": "single"})
        ens_args = argparse.Namespace(**{**vars(args), "predictor": "ensemble"})
        jobs = [("single", single_args), ("ensemble", ens_args)]
    else:
        jobs = [(predictor_mode, args)]

    all_summaries: List[Dict[str, Any]] = []
    for _label, job_args in jobs:
        runner, policy = make_predictor(job_args, arts)
        tag = str(getattr(runner, "tag", args.model))
        summary = _execute_eval(
            job_args,
            runner=runner,
            policy=policy,
            tag=tag,
            examples=examples,
            pool_info=pool_info,
            reports_dir=reports_dir,
            cell_id=args.cell,
        )
        all_summaries.append(summary)

    if len(all_summaries) > 1:
        combo_path = reports_dir / f"{args.split}_sliding_window_combined_summary.json"
        with open(combo_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"\n  Resumen combinado: {combo_path}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval ventanas deslizantes + filtros temporales/cinemáticos")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--cell", default="bin_full")
    ap.add_argument("--model", default="modelo_12")
    ap.add_argument(
        "--predictor",
        choices=["single", "ensemble", "both"],
        default="single",
        help="single | ensemble (best F1 grid) | both",
    )
    ap.add_argument(
        "--ensemble-source",
        choices=["best_f1", "best_low_fp", "auto"],
        default="best_f1",
    )
    ap.add_argument("--ensemble-models", nargs="*", default=None)
    ap.add_argument("--ensemble-rule", choices=["mean", "and"], default=None)
    ap.add_argument("--ensemble-threshold", type=float, default=0.5)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--errors-csv", default=None)

    ap.add_argument("--window-sec", type=float, default=3.0)
    ap.add_argument("--stride-sec", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS)
    ap.add_argument("--p-window-threshold", type=float, default=0.50)
    ap.add_argument("--full-clip-threshold", type=float, default=0.50)
    ap.add_argument("--min-consecutive-windows", type=int, default=2)
    ap.add_argument("--min-s-kin", type=float, default=0.42)
    ap.add_argument("--no-require-kin", action="store_true")
    ap.add_argument("--require-conceal", action="store_true")
    ap.add_argument("--post-purchase-veto-windows", type=int, default=3)
    ap.add_argument("--alarm-mode", choices=["filter_baseline", "windows_only"], default="filter_baseline")
    ap.add_argument("--windows-only", action="store_true")
    ap.add_argument("--allow-isolated-spike", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--max-fp-target", type=int, default=1)
    ap.add_argument("--min-recall-pct", type=float, default=0.0)

    args = ap.parse_args()
    return run_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
