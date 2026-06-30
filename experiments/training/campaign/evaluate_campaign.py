#!/usr/bin/env python3
"""
Evalúa modelos de campaña: métricas, barrido de umbral, ensembles y manifest FP.

Uso:
  python evaluate_campaign.py --all
  python evaluate_campaign.py --cells bin_full --export-fp-videos
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
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

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


def ensemble_grid_binary(
    model_probs: Dict[str, np.ndarray],
    y_true: np.ndarray,
    config: Dict[str, Any],
    cell_id: str,
) -> List[Dict[str, Any]]:
    """Grid AND / MEAN / CASCADE sobre pares y tríos de modelos."""
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

    for rule in rules:
        for combo in itertools.combinations(names, 2):
            for thr in thr_grid[::3]:
                _add(rule, combo, (float(thr), float(thr)))
        if rule in ("and", "mean"):
            for combo in itertools.combinations(names, 3):
                if len(rows) > int(ens_cfg.get("max_triplets", 20)) * 10:
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
    y_true_ref: Optional[np.ndarray] = None
    fp_records: List[Dict[str, Any]] = []

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
                if y_true_ref is None:
                    y_true_ref = y_true

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
    return {
        "cell_id": cell_id,
        "leaderboard": str(leader_path),
        "sweep": str(sweep_path) if sweep_rows else None,
        "ensemble": str(ens_path) if ensemble_rows else None,
        "fp_manifest": str(fp_manifest_path) if fp_records else None,
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
    args = ap.parse_args()

    config = load_merged_campaign_config(Path(args.config) if args.config else None)
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
