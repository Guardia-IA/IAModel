#!/usr/bin/env python3
"""
Evalúa modelos de campaña: métricas, barrido de umbral, ensembles y manifest FP.

Uso:
  python evaluate_campaign.py --all
  python evaluate_campaign.py --cells bin_full --export-fp-videos

  # El grid automático incluye majority (voto 2-de-3, etc.) y escribe val_top_ensembles.json (top 3).

  # Probar un ensemble puntual sin barrido completo:
  python evaluate_campaign.py --ensemble-trial --run-id campaign_20261103_230735 \\
      --ensemble-cell bin_full --split val \\
      --ensemble-models 49,53,11 --ensemble-rule majority \\
      --ensemble-high 0.55 --ensemble-low 0.45 --ensemble-min-votes 2
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

CAMPAIGN_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_DIR))

try:
    from campaign_paths import (
        load_merged_campaign_config,
        filter_cells,
        ensure_cell_dirs,
        training_plan_path,
        master_reports_dir,
    )
    from evaluate_validation import (
        evaluate_validation,
        load_split_uids,
        build_split_examples,
        _f1_pos_pct,
        _binary_block,
    )
    from train_model_operations import (
        build_model,
        build_pose_dataset_for_eval,
        _example_uid,
        _example_folder_category,
        SEED,
    )
    from export_fp_artifacts import export_fp_from_records, clip_path_from_example, example_export_paths
except ImportError as exc:
    raise SystemExit(f"Import error (¿entorno con torch?): {exc}") from exc


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    n_neg = max(int((y_true == 0).sum()), 1)
    n_pos = max(int((y_true == 1).sum()), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision_pct": 100.0 * prec,
        "recall_pct": 100.0 * rec,
        "f1_pct": 100.0 * f1,
        "fp_rate_pct": 100.0 * fp / n_neg,
        "support_pos": n_pos,
        "support_neg": n_neg,
    }


@torch.no_grad()
def collect_binary_predictions(
    model_path: Path,
    examples: List[Any],
    *,
    training_plan_path: Path,
    split_name: str = "val",
    batch_size: int = 64,
    num_workers: int = 2,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    label_to_idx = checkpoint["label_to_idx"]
    seq_len = int(checkpoint.get("seq_len", 64))
    cfg = checkpoint.get("config", {})
    arch = cfg.get("arch", "tcn")
    input_dim = int(checkpoint["input_dim"])
    num_classes = int(checkpoint.get("num_classes", len(label_to_idx)))

    ds = build_pose_dataset_for_eval(
        examples,
        label_to_idx,
        seq_len,
        dataset_split=split_name,
        checkpoint=checkpoint,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=int(num_workers))

    model = build_model(arch, input_dim, num_classes, cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    records: List[Dict[str, Any]] = []
    probs_list: List[float] = []
    margins_list: List[float] = []
    y_true: List[int] = []
    ex_iter = iter(examples)

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        prob = F.softmax(logits, dim=1)
        if num_classes == 2:
            p_pos = prob[:, 1].cpu().numpy()
            margin = (logits[:, 1] - logits[:, 0]).cpu().numpy()
        else:
            p_pos = prob[:, 0].cpu().numpy()
            margin = p_pos

        for i in range(x.size(0)):
            ex = next(ex_iter)
            yt = int(y[i].item())
            uid = _example_uid(ex)
            folder_cat = int(_example_folder_category(ex))
            path_info = example_export_paths(ex)
            rec = {
                "uid": uid,
                "true_label": yt,
                "folder_category": folder_cat,
                "prob_pos": float(p_pos[i]),
                "logit_margin": float(margin[i]),
                "pose_path": path_info["pose_path"],
                "clip_path": path_info["clip_video_path"] or path_info["clip_dir"],
                "clip_video_path": path_info["clip_video_path"],
                "clip_dir": path_info["clip_dir"],
                "clip_name": path_info["clip_name"],
                "category_str": path_info["category_str"],
                "meta_json_path": path_info["meta_json_path"],
                "user_dir": path_info["user_dir"],
                "model_path": str(model_path),
            }
            records.append(rec)
            probs_list.append(float(p_pos[i]))
            margins_list.append(float(margin[i]))
            y_true.append(yt)

    return (
        records,
        np.array(y_true, dtype=np.int64),
        np.array(probs_list, dtype=np.float64),
        np.array(margins_list, dtype=np.float64),
    )


def sweep_softmax_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    sweep: Dict[str, float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    t = float(sweep["softmax_min"])
    t_max = float(sweep["softmax_max"])
    step = float(sweep["softmax_step"])
    while t <= t_max + 1e-9:
        pred = (probs >= t).astype(np.int64)
        m = _binary_metrics(y_true, pred)
        rows.append({"decision_mode": "softmax_thr", "threshold": round(t, 4), **m})
        t += step
    return rows


def sweep_logit_margins(
    y_true: np.ndarray,
    margins: np.ndarray,
    sweep: Dict[str, float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    t = float(sweep["logit_margin_min"])
    t_max = float(sweep["logit_margin_max"])
    step = float(sweep["logit_margin_step"])
    while t <= t_max + 1e-9:
        pred = (margins >= t).astype(np.int64)
        m = _binary_metrics(y_true, pred)
        rows.append({"decision_mode": "logit_margin", "threshold": round(t, 4), **m})
        t += step
    return rows


def _clip_path(rec: Dict[str, Any]) -> str:
    return str(
        rec.get("clip_video_path")
        or rec.get("clip_dir")
        or rec.get("clip_path")
        or ""
    )


def write_path_list(path: Path, paths: List[str], *, header: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write(f"# {header}\n")
        for p in paths:
            if p:
                f.write(f"{p}\n")
    return path


def export_error_paths(
    records: List[Dict[str, Any]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    reports_dir: Path,
    logs_dir: Path,
    split: str,
    cell_id: str,
    label: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Exporta CSV de errores y listas .txt con rutas FN/FP."""
    fn_paths: List[str] = []
    fp_paths: List[str] = []
    error_rows: List[Dict[str, Any]] = []
    safe_label = label.replace("|", "+").replace("/", "_")

    for rec, yt, yp in zip(records, y_true, y_pred):
        yt_i = int(yt)
        yp_i = int(yp)
        cp = _clip_path(rec)
        if yp_i == 1 and yt_i == 0:
            fp_paths.append(cp)
            error_rows.append({**rec, "outcome": "FP", "pred_label": yp_i, "true_label": yt_i})
        elif yp_i == 0 and yt_i == 1:
            fn_paths.append(cp)
            error_rows.append({**rec, "outcome": "FN", "pred_label": yp_i, "true_label": yt_i})

    csv_path = reports_dir / f"{split}_errors_{safe_label}.csv"
    fn_report = reports_dir / f"{split}_fn_{safe_label}.txt"
    fp_report = reports_dir / f"{split}_fp_{safe_label}.txt"
    fn_log = logs_dir / f"{split}_{cell_id}_fn_{safe_label}.txt"
    fp_log = logs_dir / f"{split}_{cell_id}_fp_{safe_label}.txt"

    if error_rows:
        _write_csv(csv_path, error_rows)

    hdr = f"{cell_id} {label} split={split}"
    if meta:
        hdr += (
            f" F1={meta.get('f1_pct', '?')}% "
            f"FN={len(fn_paths)} FP={len(fp_paths)}"
        )

    write_path_list(fn_report, fn_paths, header=hdr)
    write_path_list(fp_report, fp_paths, header=hdr)
    write_path_list(fn_log, fn_paths, header=hdr)
    write_path_list(fp_log, fp_paths, header=hdr)

    return {
        "label": label,
        "fn_count": len(fn_paths),
        "fp_count": len(fp_paths),
        "errors_csv": str(csv_path) if error_rows else None,
        "fn_paths_txt": str(fn_report),
        "fp_paths_txt": str(fp_report),
        "fn_paths_log": str(fn_log),
        "fp_paths_log": str(fp_log),
        **(meta or {}),
    }


def pick_best_sweep_row(
    sweep_rows: List[Dict[str, Any]],
    model_name: str,
    *,
    decision_mode: str = "softmax_thr",
) -> Optional[Dict[str, Any]]:
    rows = [
        r
        for r in sweep_rows
        if r.get("model") == model_name and r.get("decision_mode") == decision_mode
    ]
    if not rows:
        return None
    return max(rows, key=lambda r: float(r.get("f1_pct", 0)))


def normalize_model_stems(models: List[str]) -> List[str]:
    out: List[str] = []
    for raw in models:
        s = str(raw).strip()
        if not s:
            continue
        if s.endswith(".pt"):
            s = Path(s).stem
        if not s.startswith("modelo_"):
            if s.isdigit():
                s = f"modelo_{int(s):02d}"
            else:
                s = f"modelo_{s}"
        out.append(s)
    return out


def ensemble_majority_predict(
    arrays: List[np.ndarray],
    *,
    high: float,
    low: float,
    min_agree: int,
    tie_pred: int = 0,
) -> np.ndarray:
    """Voto: robo si >=min_agree tienen P>=high; no robo si >=min_agree tienen P<=low; si no, tie_pred."""
    if not arrays:
        raise ValueError("ensemble_majority_predict requiere al menos un modelo")
    stack = np.stack(arrays, axis=0)
    n = stack.shape[1]
    min_agree = max(1, int(min_agree))
    high_votes = (stack >= float(high)).sum(axis=0)
    low_votes = (stack <= float(low)).sum(axis=0)
    pred = np.full(n, int(tie_pred), dtype=np.int64)
    pred[low_votes >= min_agree] = 0
    pred[high_votes >= min_agree] = 1
    return pred


def collect_model_probs_for_cell(
    cell: Dict[str, Any],
    config: Dict[str, Any],
    *,
    split: str,
    model_stems: List[str],
    run_id: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[Any]]:
    """Inferencia solo sobre los modelos pedidos; devuelve y_true, probs por stem y examples."""
    cell_id = cell["id"]
    arts = ensure_cell_dirs(cell_id, run_id=run_id)
    plan_path = training_plan_path(cell_id, run_id=run_id)
    if not plan_path.is_file():
        raise FileNotFoundError(f"Falta plan: {plan_path}")

    split_uids, split_meta = load_split_uids(split_name=split, training_plan_path=plan_path)
    split_meta["split_name"] = split
    split_meta["split_uids_all"] = {
        k: [str(x) for x in v] for k, v in json.loads(plan_path.read_text()).get("split_uids", {}).items()
    }
    examples, _pool = build_split_examples(
        split_uids=split_uids,
        split_meta=split_meta,
        pose_source=cell["pose_source"],
        single_user_only=bool(config.get("single_user_only", True)),
        task=cell["task"],
    )
    if cell["task"] != "binary":
        raise ValueError(f"--ensemble-trial solo aplica a celdas binarias (recibido {cell_id!r})")

    model_probs: Dict[str, np.ndarray] = {}
    y_true_ref: Optional[np.ndarray] = None
    for stem in model_stems:
        mp = arts["models_dir"] / f"{stem}.pt"
        if not mp.is_file():
            raise FileNotFoundError(f"No existe {mp}")
        _records, y_true, probs, _margins = collect_binary_predictions(
            mp, examples, training_plan_path=plan_path, split_name=split
        )
        model_probs[stem] = probs
        if y_true_ref is None:
            y_true_ref = y_true
    assert y_true_ref is not None
    return y_true_ref, model_probs, examples


def run_ensemble_trial(
    *,
    cell: Dict[str, Any],
    config: Dict[str, Any],
    split: str,
    model_stems: List[str],
    rule: str,
    run_id: Optional[str],
    mean_threshold: float = 0.5,
    majority_high: float = 0.55,
    majority_low: float = 0.45,
    majority_min_votes: int = 2,
    majority_tie: int = 0,
) -> Dict[str, Any]:
    y_true, model_probs, _examples = collect_model_probs_for_cell(
        cell, config, split=split, model_stems=model_stems, run_id=run_id
    )
    arrays = [model_probs[n] for n in model_stems]
    rule = str(rule).strip().lower()

    if rule == "mean":
        mean_p = np.mean(np.stack(arrays, axis=0), axis=0)
        pred = (mean_p >= float(mean_threshold)).astype(np.int64)
        thresholds_str = f"{mean_threshold:.3f}"
    elif rule == "and":
        thr = float(mean_threshold)
        mask = np.ones(len(y_true), dtype=bool)
        for arr in arrays:
            mask &= arr >= thr
        pred = mask.astype(np.int64)
        thresholds_str = "|".join(f"{thr:.3f}" for _ in model_stems)
    elif rule == "majority":
        pred = ensemble_majority_predict(
            arrays,
            high=majority_high,
            low=majority_low,
            min_agree=majority_min_votes,
            tie_pred=majority_tie,
        )
        thresholds_str = (
            f"high={majority_high:.3f}|low={majority_low:.3f}"
            f"|min_votes={majority_min_votes}|tie={majority_tie}"
        )
    else:
        raise ValueError(f"Regla no soportada: {rule!r} (mean, and, majority)")

    m = _binary_metrics(y_true, pred)
    n_neg = int(m["support_neg"])
    n_pos = int(m["support_pos"])
    summary = {
        "cell_id": cell["id"],
        "split": split,
        "run_id": run_id,
        "models": model_stems,
        "rule": rule,
        "decision_mode": f"ensemble_{rule}",
        "thresholds": thresholds_str,
        **m,
        "fn_count": m["fn"],
        "fp_count": m["fp"],
        "alarms_of": n_neg,
        "missed_robberies_of": n_pos,
    }

    arts = ensure_cell_dirs(cell["id"], run_id=run_id)
    out_path = arts["reports_dir"] / f"{split}_ensemble_trial_{rule}_{'+'.join(model_stems)}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    summary["report_path"] = str(out_path)

    print(f"\n=== Ensemble trial — {cell['id']} split={split} ===")
    print(f"Modelos: {' + '.join(model_stems)} | regla: {rule}")
    print(f"Umbrales: {thresholds_str}")
    print(
        f"F1 {m['f1_pct']:.2f}% | recall {m['recall_pct']:.2f}% | "
        f"FP {m['fp_rate_pct']:.2f}%"
    )
    print(
        f"Alarmas {m['fp']} de {n_neg} | robos no vistos {m['fn']} de {n_pos} "
        f"(TP {m['tp']} TN {m['tn']})"
    )
    print(f"Informe: {out_path}")
    return summary


def _ensemble_row_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("decision_mode", "")),
        str(row.get("models", "")),
        str(row.get("thresholds", "")),
    )


def rank_ensemble_rows(
    ensemble_rows: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Orden operativo: recall≥objetivo, mínimo FP, máximo F1; luego rellena por F1."""
    if not ensemble_rows:
        return []
    target_rec = float(config.get("target_recall_pct", 60.0))
    target_fp = float(config.get("target_fp_rate_pct", 0.01))

    def op_key(r: Dict[str, Any]) -> Tuple[float, float]:
        return (float(r.get("fp_rate_pct", 999)), -float(r.get("f1_pct", 0)))

    meets_recall = [r for r in ensemble_rows if float(r.get("recall_pct", 0)) >= target_rec]
    meets_both = [r for r in meets_recall if float(r.get("fp_rate_pct", 999)) <= target_fp]
    pool = sorted(meets_both or meets_recall, key=op_key) if meets_recall else []
    by_f1 = sorted(
        ensemble_rows,
        key=lambda r: (-float(r.get("f1_pct", 0)), float(r.get("fp_rate_pct", 999))),
    )

    ranked: List[Dict[str, Any]] = []
    seen: set = set()
    for r in pool + by_f1:
        key = _ensemble_row_key(r)
        if key in seen:
            continue
        seen.add(key)
        ranked.append(r)
    return ranked


def pick_top_ensemble_rows(
    ensemble_rows: List[Dict[str, Any]],
    config: Dict[str, Any],
    n: int = 3,
) -> List[Dict[str, Any]]:
    ens_cfg = config.get("ensemble") or {}
    top_n = max(1, int(ens_cfg.get("top_n", n)))
    return rank_ensemble_rows(ensemble_rows, config)[:top_n]


def pick_best_ensemble_row(
    ensemble_rows: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    top = pick_top_ensemble_rows(ensemble_rows, config, n=1)
    return top[0] if top else None


def ensemble_predict_from_row(
    model_probs: Dict[str, np.ndarray],
    row: Dict[str, Any],
    config: Dict[str, Any],
) -> np.ndarray:
    ens_cfg = config.get("ensemble") or {}
    model_names = [m.strip() for m in str(row.get("models", "")).split("|") if m.strip()]
    rule = str(row.get("decision_mode", "")).replace("ensemble_", "")
    thr_parts = [float(t) for t in str(row.get("thresholds", "")).split("|") if t.strip()]
    arrays = [model_probs[n] for n in model_names]
    n = len(arrays[0])

    if rule == "and":
        mask = np.ones(n, dtype=bool)
        for arr, thr in zip(arrays, thr_parts):
            mask &= arr >= thr
        return mask.astype(np.int64)
    if rule == "mean":
        mean_p = np.mean(np.stack(arrays, axis=0), axis=0)
        thr = thr_parts[0] if thr_parts else 0.5
        return (mean_p >= thr).astype(np.int64)
    if rule == "majority":
        ens_cfg = config.get("ensemble") or {}
        high = float(ens_cfg.get("majority_high", 0.55))
        low = float(ens_cfg.get("majority_low", 0.45))
        min_votes = int(ens_cfg.get("majority_min_votes", 2))
        tie = int(ens_cfg.get("majority_tie_pred", 0))
        thr_raw = str(row.get("thresholds", ""))
        for part in thr_raw.split("|"):
            if part.startswith("high="):
                high = float(part.split("=", 1)[1])
            elif part.startswith("low="):
                low = float(part.split("=", 1)[1])
            elif part.startswith("min_votes="):
                min_votes = int(part.split("=", 1)[1])
            elif part.startswith("tie="):
                tie = int(part.split("=", 1)[1])
        return ensemble_majority_predict(
            arrays, high=high, low=low, min_agree=min_votes, tie_pred=tie
        )
    if rule == "cascade":
        low = float(ens_cfg.get("cascade_low", 0.4))
        high = float(ens_cfg.get("cascade_high", 0.55))
        pred = np.zeros(n, dtype=np.int64)
        for i in range(n):
            if arrays[0][i] >= low and arrays[1][i] >= high:
                pred[i] = 1
        return pred
    raise ValueError(f"Regla ensemble no soportada: {rule!r}")


def write_top_ensembles_spec(
    reports_dir: Path,
    split: str,
    rows: List[Dict[str, Any]],
    *,
    selection: str = "ranked",
) -> Path:
    specs = []
    for rank, row in enumerate(rows, start=1):
        rule = str(row.get("decision_mode", "")).replace("ensemble_", "")
        thr_raw = str(row.get("thresholds", ""))
        thr_parts = [float(t) for t in thr_raw.split("|") if t.strip() and "=" not in t]
        models = [m.strip() for m in str(row.get("models", "")).split("|") if m.strip()]
        specs.append({
            "rank": rank,
            "split": split,
            "models": models,
            "rule": rule,
            "threshold": thr_parts[0] if rule == "mean" and thr_parts else None,
            "thresholds": thr_raw,
            "decision_mode": row.get("decision_mode"),
            "f1_pct": row.get("f1_pct"),
            "recall_pct": row.get("recall_pct"),
            "fp_rate_pct": row.get("fp_rate_pct"),
            "fn": row.get("fn"),
            "fp": row.get("fp"),
        })
    path = reports_dir / f"{split}_top_ensembles.json"
    payload = {"split": split, "selection": selection, "ensembles": specs}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def write_best_ensemble_spec(
    reports_dir: Path,
    split: str,
    row: Dict[str, Any],
) -> Path:
    rule = str(row.get("decision_mode", "")).replace("ensemble_", "")
    thr_parts = [float(t) for t in str(row.get("thresholds", "")).split("|") if t.strip()]
    models = [m.strip() for m in str(row.get("models", "")).split("|") if m.strip()]
    spec = {
        "split": split,
        "models": models,
        "rule": rule,
        "threshold": thr_parts[0] if rule == "mean" and thr_parts else thr_parts,
        "thresholds": thr_parts,
        "decision_mode": row.get("decision_mode"),
        "f1_pct": row.get("f1_pct"),
        "recall_pct": row.get("recall_pct"),
        "fp_rate_pct": row.get("fp_rate_pct"),
        "fn": row.get("fn"),
        "fp": row.get("fp"),
    }
    path = reports_dir / f"{split}_best_ensemble.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def load_best_ensemble_spec(reports_dir: Path, split: str) -> Optional[Dict[str, Any]]:
    path = reports_dir / f"{split}_best_ensemble.json"
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensemble_grid_binary(
    model_probs: Dict[str, np.ndarray],
    y_true: np.ndarray,
    config: Dict[str, Any],
    cell_id: str,
) -> List[Dict[str, Any]]:
    """Grid AND / MEAN / CASCADE / MAJORITY sobre pares y tríos de modelos."""
    ens_cfg = config.get("ensemble") or {}
    rules = list(ens_cfg.get("rules") or ["and", "mean"])
    names = sorted(model_probs.keys())
    rows: List[Dict[str, Any]] = []
    sweep = config.get("threshold_sweep") or {}
    thr_grid = np.arange(
        float(sweep.get("softmax_min", 0.5)),
        float(sweep.get("softmax_max", 0.95)) + 1e-9,
        float(sweep.get("softmax_step", 0.05)),
    )
    max_triplet_rows = int(ens_cfg.get("max_triplets", 20)) * 10
    tie_pred = int(ens_cfg.get("majority_tie_pred", 0))
    high_grid = [float(x) for x in (ens_cfg.get("majority_high_grid") or thr_grid[::4].tolist())]
    low_grid = [float(x) for x in (ens_cfg.get("majority_low_grid") or [0.4, 0.45, 0.48])]
    min_votes_list = [int(x) for x in (ens_cfg.get("majority_min_votes") or [2])]

    def _add(rule: str, combo: Tuple[str, ...], thr_vals: Tuple[float, ...]) -> None:
        arrays = [model_probs[n] for n in combo]
        if rule == "and":
            mask = np.ones(len(y_true), dtype=bool)
            for arr, thr in zip(arrays, thr_vals):
                mask &= arr >= thr
            pred = mask.astype(np.int64)
        elif rule == "mean":
            mean_p = np.mean(np.stack(arrays, axis=0), axis=0)
            pred = (mean_p >= thr_vals[0]).astype(np.int64)
        elif rule == "cascade":
            low = float(ens_cfg.get("cascade_low", 0.4))
            high = float(ens_cfg.get("cascade_high", 0.55))
            pred = np.zeros(len(y_true), dtype=np.int64)
            for i in range(len(y_true)):
                if arrays[0][i] >= low and arrays[1][i] >= high:
                    pred[i] = 1
        else:
            return
        m = _binary_metrics(y_true, pred)
        rows.append({
            "cell_id": cell_id,
            "decision_mode": f"ensemble_{rule}",
            "models": "|".join(combo),
            "thresholds": "|".join(f"{t:.3f}" for t in thr_vals),
            **m,
        })

    def _add_majority(
        combo: Tuple[str, ...],
        *,
        high: float,
        low: float,
        min_votes: int,
    ) -> None:
        if low >= high:
            return
        if min_votes > len(combo):
            return
        arrays = [model_probs[n] for n in combo]
        pred = ensemble_majority_predict(
            arrays,
            high=high,
            low=low,
            min_agree=min_votes,
            tie_pred=tie_pred,
        )
        m = _binary_metrics(y_true, pred)
        thr_str = (
            f"high={high:.3f}|low={low:.3f}|min_votes={min_votes}|tie={tie_pred}"
        )
        rows.append({
            "cell_id": cell_id,
            "decision_mode": "ensemble_majority",
            "models": "|".join(combo),
            "thresholds": thr_str,
            **m,
        })

    for rule in rules:
        if rule == "majority":
            sampled_high = high_grid[::2] if len(high_grid) > 4 else high_grid
            sampled_low = low_grid[::1] if len(low_grid) <= 4 else low_grid[::2]
            for combo in itertools.combinations(names, 2):
                for high in sampled_high:
                    for low in sampled_low:
                        for mv in min_votes_list:
                            _add_majority(combo, high=high, low=low, min_votes=mv)
            for combo in itertools.combinations(names, 3):
                if len(rows) > max_triplet_rows:
                    break
                highs = sampled_high[::2] if len(sampled_high) > 3 else sampled_high
                lows = sampled_low[::2] if len(sampled_low) > 3 else sampled_low
                for high in highs:
                    for low in lows:
                        for mv in min_votes_list:
                            _add_majority(combo, high=high, low=low, min_votes=mv)
            continue
        for combo in itertools.combinations(names, 2):
            if rule == "cascade":
                _add(rule, combo, (0.0, 0.0))
                continue
            for thr in thr_grid[::3]:
                _add(rule, combo, (float(thr), float(thr)))
        if rule in ("and", "mean"):
            for combo in itertools.combinations(names, 3):
                if len(rows) > max_triplet_rows:
                    break
                for thr in thr_grid[::4]:
                    thrs = (float(thr),) * len(combo)
                    _add(rule, combo, thrs)
    return rows


def evaluate_cell(
    cell: Dict[str, Any],
    config: Dict[str, Any],
    *,
    split: str = "val",
    export_fp: bool = False,
    max_fp_export: int = 100,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    cell_id = cell["id"]
    arts = ensure_cell_dirs(cell_id, run_id=run_id)
    models_dir = arts["models_dir"]
    reports_dir = arts["reports_dir"]
    plan_path = training_plan_path(cell_id, run_id=run_id)

    if not plan_path.is_file():
        raise FileNotFoundError(f"Falta plan: {plan_path}")

    model_paths = sorted(models_dir.glob("modelo_*.pt"))
    if not model_paths:
        raise FileNotFoundError(f"No hay modelos en {models_dir}")

    split_uids, split_meta = load_split_uids(split_name=split, training_plan_path=plan_path)
    split_meta["split_name"] = split
    split_meta["split_uids_all"] = {
        k: [str(x) for x in v] for k, v in json.loads(plan_path.read_text()).get("split_uids", {}).items()
    }

    examples, pool_info = build_split_examples(
        split_uids=split_uids,
        split_meta=split_meta,
        pose_source=cell["pose_source"],
        single_user_only=bool(config.get("single_user_only", True)),
        task=cell["task"],
    )

    per_model_reports: List[Dict[str, Any]] = []
    sweep_rows: List[Dict[str, Any]] = []
    leaderboard_rows: List[Dict[str, Any]] = []
    model_probs: Dict[str, np.ndarray] = {}
    model_records: Dict[str, List[Dict[str, Any]]] = {}
    y_true_ref: Optional[np.ndarray] = None
    base_records_ref: Optional[List[Dict[str, Any]]] = None
    fp_records: List[Dict[str, Any]] = []
    per_model_best: List[Dict[str, Any]] = []
    error_exports: List[Dict[str, Any]] = []
    logs_dir = arts["logs_dir"]

    for mp in model_paths:
        exp_name = mp.stem
        try:
            report = evaluate_validation(
                mp,
                split_name=split,
                training_plan_path=plan_path,
                pose_source=cell["pose_source"],
                single_user_only=bool(config.get("single_user_only", True)),
                quiet=True,
            )
            per_model_reports.append(report)

            if cell["task"] == "binary":
                b_argmax = _binary_block(report, "softmax_argmax")
                leaderboard_rows.append({
                    "cell_id": cell_id,
                    "task": cell["task"],
                    "pose_source": cell["pose_source"],
                    "model": mp.name,
                    "decision_mode": "softmax_argmax",
                    "threshold": "",
                    "f1_pct": _f1_pos_pct(report, "softmax_argmax"),
                    "recall_pct": b_argmax.get("recall_robbery_pct", 0),
                    "fp_rate_pct": b_argmax.get("false_positive_rate_pct", 0),
                    "accuracy_pct": 100 * float(report.get("accuracy", 0)),
                })
                for mode in ("softmax_threshold", "logit_margin"):
                    b = _binary_block(report, mode)
                    if b:
                        leaderboard_rows.append({
                            "cell_id": cell_id,
                            "task": cell["task"],
                            "pose_source": cell["pose_source"],
                            "model": mp.name,
                            "decision_mode": mode,
                            "threshold": report.get("binary_softmax_threshold") if mode == "softmax_threshold" else report.get("binary_logit_margin"),
                            "f1_pct": _f1_pos_pct(report, mode),
                            "recall_pct": b.get("recall_robbery_pct", 0),
                            "fp_rate_pct": b.get("false_positive_rate_pct", 0),
                            "accuracy_pct": 100 * float(report.get("accuracy", 0)),
                        })

                records, y_true, probs, margins = collect_binary_predictions(
                    mp, examples, training_plan_path=plan_path, split_name=split
                )
                model_probs[exp_name] = probs
                model_records[exp_name] = records
                if y_true_ref is None:
                    y_true_ref = y_true
                    base_records_ref = records

                for row in sweep_softmax_thresholds(y_true, probs, config.get("threshold_sweep", {})):
                    row.update({"cell_id": cell_id, "model": mp.name})
                    sweep_rows.append(row)
                for row in sweep_logit_margins(y_true, margins, config.get("threshold_sweep", {})):
                    row.update({"cell_id": cell_id, "model": mp.name})
                    sweep_rows.append(row)

                for rec in records:
                    if rec["true_label"] == 0 and rec["prob_pos"] >= 0.5:
                        fp_records.append({**rec, "cell_id": cell_id, "exp_name": exp_name})

            else:
                rm = report.get("robbery_class_metrics") or {}
                leaderboard_rows.append({
                    "cell_id": cell_id,
                    "task": cell["task"],
                    "pose_source": cell["pose_source"],
                    "model": mp.name,
                    "decision_mode": "argmax_c6",
                    "threshold": "",
                    "f1_pct": float(rm.get("f1_pct", 0)),
                    "recall_pct": float(rm.get("recall_pct", 0)),
                    "fp_rate_pct": float(rm.get("false_positive_rate_pct", 0)),
                    "accuracy_pct": 100 * float(report.get("accuracy", 0)),
                })
        except Exception as exc:
            per_model_reports.append({"model_path": str(mp), "error": str(exc)})

    ensemble_rows: List[Dict[str, Any]] = []
    best_ensemble_info: Optional[Dict[str, Any]] = None
    if cell["task"] == "binary" and model_probs and y_true_ref is not None:
        ensemble_rows = ensemble_grid_binary(model_probs, y_true_ref, config, cell_id)
        for row in ensemble_rows:
            leaderboard_rows.append({
                "cell_id": cell_id,
                "task": cell["task"],
                "pose_source": cell["pose_source"],
                "model": row.get("models", ""),
                "decision_mode": row.get("decision_mode", ""),
                "threshold": row.get("thresholds", ""),
                "f1_pct": row.get("f1_pct", 0),
                "recall_pct": row.get("recall_pct", 0),
                "fp_rate_pct": row.get("fp_rate_pct", 0),
                "accuracy_pct": "",
            })

        for mp in model_paths:
            exp_name = mp.stem
            if exp_name not in model_probs or exp_name not in model_records:
                continue
            best_row = pick_best_sweep_row(sweep_rows, mp.name)
            if best_row is None:
                continue
            thr = float(best_row["threshold"])
            y_pred = (model_probs[exp_name] >= thr).astype(np.int64)
            err = export_error_paths(
                model_records[exp_name],
                y_true_ref,
                y_pred,
                reports_dir=reports_dir,
                logs_dir=logs_dir,
                split=split,
                cell_id=cell_id,
                label=exp_name,
                meta={
                    "model": mp.name,
                    "decision_mode": "softmax_thr",
                    "threshold": thr,
                    "f1_pct": best_row.get("f1_pct"),
                    "recall_pct": best_row.get("recall_pct"),
                    "fp_rate_pct": best_row.get("fp_rate_pct"),
                },
            )
            per_model_best.append(err)
            error_exports.append(err)

        top_ens_rows = pick_top_ensemble_rows(ensemble_rows, config)
        best_ens_row = top_ens_rows[0] if top_ens_rows else None
        if top_ens_rows:
            write_top_ensembles_spec(reports_dir, split, top_ens_rows)
        if best_ens_row is not None and base_records_ref is not None:
            y_pred_ens = ensemble_predict_from_row(model_probs, best_ens_row, config)
            ens_label = (
                f"ensemble_{best_ens_row.get('decision_mode', '').replace('ensemble_', '')}_"
                f"{best_ens_row.get('models', '')}_t{best_ens_row.get('thresholds', '')}"
            ).replace("|", "+")
            err_ens = export_error_paths(
                base_records_ref,
                y_true_ref,
                y_pred_ens,
                reports_dir=reports_dir,
                logs_dir=logs_dir,
                split=split,
                cell_id=cell_id,
                label=ens_label,
                meta={
                    "models": best_ens_row.get("models"),
                    "decision_mode": best_ens_row.get("decision_mode"),
                    "thresholds": best_ens_row.get("thresholds"),
                    "f1_pct": best_ens_row.get("f1_pct"),
                    "recall_pct": best_ens_row.get("recall_pct"),
                    "fp_rate_pct": best_ens_row.get("fp_rate_pct"),
                },
            )
            best_spec_path = write_best_ensemble_spec(reports_dir, split, best_ens_row)
            best_ensemble_info = {
                **err_ens,
                "best_ensemble_json": str(best_spec_path),
                "models": best_ens_row.get("models"),
                "decision_mode": best_ens_row.get("decision_mode"),
                "thresholds": best_ens_row.get("thresholds"),
            }
            error_exports.append(err_ens)

    per_model_best_path = reports_dir / f"{split}_per_model_best.json"
    if per_model_best:
        with open(per_model_best_path, "w", encoding="utf-8") as f:
            json.dump(per_model_best, f, indent=2, ensure_ascii=False)
            f.write("\n")

    out_eval = reports_dir / f"{split}_eval_summary.json"
    payload = {
        "cell_id": cell_id,
        "split": split,
        "pool_info": pool_info,
        "models_evaluated": len(per_model_reports),
        "reports": per_model_reports,
        "seed": SEED,
    }
    with open(out_eval, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    sweep_path = reports_dir / f"{split}_decision_sweep.csv"
    if sweep_rows:
        _write_csv(sweep_path, sweep_rows)

    leader_path = reports_dir / f"{split}_leaderboard.csv"
    _write_csv(leader_path, leaderboard_rows)

    ens_path = reports_dir / f"{split}_ensemble_grid.csv"
    if ensemble_rows:
        _write_csv(ens_path, ensemble_rows)

    fp_manifest_path = reports_dir / f"{split}_fp_manifest.csv"
    if fp_records:
        fp_records.sort(key=lambda r: -r["prob_pos"])
        _write_csv(fp_manifest_path, fp_records[: max_fp_export * len(model_paths)])
        if export_fp:
            export_fp_from_records(
                fp_records[:max_fp_export],
                arts["fp_clips_dir"],
                cell_id=cell_id,
            )

    print(f"  [OK] {cell_id}: leaderboard → {leader_path}")
    if cell["task"] == "binary" and per_model_best:
        print(f"  [{cell_id}] Resultados por modelo (mejor umbral softmax en {split}):")
        for pm in sorted(per_model_best, key=lambda r: -float(r.get("f1_pct") or 0)):
            print(
                f"    {pm.get('model', pm.get('label'))} @ {pm.get('threshold')}: "
                f"F1={float(pm.get('f1_pct') or 0):.1f}% "
                f"FN={pm.get('fn_count')} FP={pm.get('fp_count')}"
            )
            print(f"      robos no detectados → {pm.get('fn_paths_log')}")
            print(f"      falsos positivos    → {pm.get('fp_paths_log')}")
    if ensemble_rows and cell["task"] == "binary":
        top_ens_rows = pick_top_ensemble_rows(ensemble_rows, config)
        if top_ens_rows:
            top_path = reports_dir / f"{split}_top_ensembles.json"
            print(f"  [{cell_id}] Top {len(top_ens_rows)} ensembles ({split}) → {top_path}")
            for row in top_ens_rows:
                print(
                    f"    · {row.get('decision_mode')} {row.get('models')} "
                    f"@ {row.get('thresholds')} — "
                    f"F1={float(row.get('f1_pct') or 0):.1f}% "
                    f"rec={float(row.get('recall_pct') or 0):.1f}% "
                    f"FN={row.get('fn')} FP={row.get('fp')}"
                )
    if best_ensemble_info:
        print(
            f"  [{cell_id}] Mejor ensemble operativo ({split}, export FP/FN): "
            f"{best_ensemble_info.get('decision_mode')} "
            f"{best_ensemble_info.get('models')} @ {best_ensemble_info.get('thresholds')} — "
            f"F1={float(best_ensemble_info.get('f1_pct') or 0):.1f}% "
            f"FN={best_ensemble_info.get('fn_count')} FP={best_ensemble_info.get('fp_count')}"
        )
        print(f"      spec → {best_ensemble_info.get('best_ensemble_json')}")
        print(f"      robos no detectados → {best_ensemble_info.get('fn_paths_log')}")
        print(f"      falsos positivos    → {best_ensemble_info.get('fp_paths_log')}")

    return {
        "cell_id": cell_id,
        "leaderboard": str(leader_path),
        "sweep": str(sweep_path) if sweep_rows else None,
        "ensemble": str(ens_path) if ensemble_rows else None,
        "best_ensemble": best_ensemble_info.get("best_ensemble_json") if best_ensemble_info else None,
        "top_ensembles": str(reports_dir / f"{split}_top_ensembles.json")
        if ensemble_rows and cell["task"] == "binary"
        else None,
        "per_model_best": str(per_model_best_path) if per_model_best else None,
        "fp_manifest": str(fp_manifest_path) if fp_records else None,
        "error_exports": error_exports,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluación de campaña")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--split", type=str, default="val", choices=["val", "test"])
    ap.add_argument("--export-fp-videos", action="store_true")
    ap.add_argument("--max-fp-export", type=int, default=100)
    ap.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Evaluar modelos bajo artifacts/runs/<run-id>/",
    )
    ap.add_argument(
        "--data-augmentation",
        action="store_true",
        help="Lo aplica el preflight. La evaluación usa el plan ya escrito.",
    )
    ap.add_argument(
        "--learning-curve",
        "--prediction",
        dest="learning_curve",
        action="store_true",
        help="Evalúa cada tamaño de la curva (run_id lc_<N>)",
    )
    ap.add_argument(
        "--train-sizes",
        nargs="+",
        default=None,
        metavar="N|max",
        help="Tamaños train para --learning-curve (entero o 'max')",
    )
    ap.add_argument(
        "--ensemble-trial",
        action="store_true",
        help="Evalúa un ensemble concreto (mean / and / majority) sin barrido completo.",
    )
    ap.add_argument(
        "--ensemble-cell",
        type=str,
        default=None,
        help="Celda binaria (p. ej. bin_full). Obligatorio con --ensemble-trial.",
    )
    ap.add_argument(
        "--ensemble-models",
        type=str,
        default=None,
        help="IDs separados por coma: 49,53,11 o modelo_49|modelo_53.",
    )
    ap.add_argument(
        "--ensemble-rule",
        type=str,
        default="mean",
        choices=["mean", "and", "majority"],
        help="Regla de fusión (default mean).",
    )
    ap.add_argument(
        "--ensemble-threshold",
        type=float,
        default=0.5,
        help="Umbral para mean (y and si todos comparten umbral).",
    )
    ap.add_argument(
        "--ensemble-high",
        type=float,
        default=0.55,
        help="Con majority: P(robo) alta si >= este valor.",
    )
    ap.add_argument(
        "--ensemble-low",
        type=float,
        default=0.45,
        help="Con majority: P(robo) baja si <= este valor.",
    )
    ap.add_argument(
        "--ensemble-min-votes",
        type=int,
        default=2,
        help="Con majority: mínimo de modelos que deben coincidir.",
    )
    ap.add_argument(
        "--ensemble-tie",
        type=int,
        default=0,
        choices=[0, 1],
        help="Con majority: predicción si no hay mayoría clara (0=no robo, 1=robo).",
    )
    args = ap.parse_args()

    config = load_merged_campaign_config(Path(args.config) if args.config else None)

    if args.ensemble_trial:
        if not args.ensemble_cell or not args.ensemble_models:
            print("Con --ensemble-trial indica --ensemble-cell y --ensemble-models.", file=sys.stderr)
            return 1
        raw = args.ensemble_models.replace("|", ",")
        stems = normalize_model_stems([p.strip() for p in raw.split(",") if p.strip()])
        if len(stems) < 2:
            print("Indica al menos dos modelos en --ensemble-models.", file=sys.stderr)
            return 1
        cells = filter_cells(config, [args.ensemble_cell])
        if not cells:
            print(f"Celda desconocida: {args.ensemble_cell}", file=sys.stderr)
            return 1
        cell = cells[0]
        run_id = str(args.run_id).strip() if args.run_id else None
        try:
            run_ensemble_trial(
                cell=cell,
                config=config,
                split=args.split,
                model_stems=stems,
                rule=args.ensemble_rule,
                run_id=run_id,
                mean_threshold=float(args.ensemble_threshold),
                majority_high=float(args.ensemble_high),
                majority_low=float(args.ensemble_low),
                majority_min_votes=int(args.ensemble_min_votes),
                majority_tie=int(args.ensemble_tie),
            )
        except Exception as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        return 0

    if args.learning_curve:
        from learning_curve_utils import (
            get_learning_curve_train_sizes,
            resolve_learning_curve_cells,
            run_id_for_train_size,
        )

        try:
            cells = resolve_learning_curve_cells(config, args.cells)
            train_sizes = get_learning_curve_train_sizes(
                cli_sizes=args.train_sizes,
                config=config,
            )
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

        for cell in cells:
            cid = cell["id"]
            for n in train_sizes:
                rid = run_id_for_train_size(n)
                print(f"\n{'#' * 80}\n# LC eval — {cid} size={n} run_id={rid}\n{'#' * 80}")
                try:
                    evaluate_cell(
                        cell,
                        config,
                        split=args.split,
                        export_fp=args.export_fp_videos,
                        max_fp_export=args.max_fp_export,
                        run_id=rid,
                    )
                except Exception as exc:
                    print(f"  ERROR {cid}: {exc}", file=sys.stderr)
        return 0

    run_id = str(args.run_id).strip() if args.run_id else None
    cells = filter_cells(config, None if args.all or not args.cells else args.cells)
    if not cells:
        print("Indica --all o --cells.", file=sys.stderr)
        return 1

    summary: List[Dict[str, Any]] = []
    print(f"\n=== Eval campaña — {len(cells)} celdas, split={args.split} ===")
    if run_id:
        print(f"Run ID: {run_id}\n")
    else:
        print()
    for cell in cells:
        try:
            row = evaluate_cell(
                cell,
                config,
                split=args.split,
                export_fp=args.export_fp_videos,
                max_fp_export=args.max_fp_export,
                run_id=run_id,
            )
            summary.append(row)
        except Exception as exc:
            print(f"  ERROR {cell['id']}: {exc}", file=sys.stderr)
            summary.append({"cell_id": cell["id"], "error": str(exc)})

    master = master_reports_dir(run_id) / f"{args.split}_eval_cells.json"
    with open(master, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\nResumen master: {master}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
