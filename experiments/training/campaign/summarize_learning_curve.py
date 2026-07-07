#!/usr/bin/env python3
"""
Comparativa de curva de aprendizaje: F1, FP, FN y desglose FP por categoría.

Lee los CSV de export_ensemble_fp (--outcomes errors) por cada run_id lc_<N>.

Uso:
  python summarize_learning_curve.py
  python summarize_learning_curve.py --train-sizes 3000 6500 10000
  python summarize_learning_curve.py --split val --cell bin_full_hardened
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

CAMPAIGN_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_DIR))

try:
    from campaign_paths import (
        load_merged_campaign_config,
        artifacts_root,
        master_reports_dir,
    )
    from learning_curve_utils import (
        get_learning_curve_train_sizes,
        resolve_learning_curve_cells,
        load_learning_curve_cell_ids,
        run_id_for_train_size,
        resolve_ensemble_settings,
        LEARNING_CURVE_MASTER_RUN_ID,
    )
except ImportError as exc:
    raise SystemExit(f"Import error: {exc}") from exc


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _find_ensemble_csv(reports_dir: Path, split: str, rule: str, models: Sequence[str], threshold: float) -> Optional[Path]:
    model_tag = "+".join(m.replace(".pt", "") for m in models)
    pattern = f"{split}_ensemble_fp_{rule}_{model_tag}_t{threshold:.2f}.csv"
    direct = reports_dir / pattern
    if direct.is_file():
        return direct
    candidates = sorted(reports_dir.glob(f"{split}_ensemble_fp_{rule}_*_t{threshold:.2f}.csv"))
    return candidates[0] if candidates else None


def _float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def analyze_size_run(
    *,
    train_size: int,
    run_id: str,
    cell_id: str,
    split: str,
    rule: str,
    models: Sequence[str],
    threshold: float,
) -> Dict[str, Any]:
    reports_dir = artifacts_root(run_id) / "reports" / cell_id
    summary_json = reports_dir / f"{split}_ensemble_fp_summary.json"
    csv_path = _find_ensemble_csv(reports_dir, split, rule, models, threshold)

    metrics: Dict[str, Any] = {}
    if summary_json.is_file():
        with open(summary_json, "r", encoding="utf-8") as f:
            summary = json.load(f)
        metrics = dict(summary.get("metrics") or {})

    rows = _read_csv(csv_path) if csv_path else []
    fp_rows = [r for r in rows if r.get("outcome") == "FP"]
    fn_rows = [r for r in rows if r.get("outcome") == "FN"]

    fp_by_cat = Counter(str(r.get("folder_category", "?")) for r in fp_rows)
    fn_by_cat = Counter(str(r.get("folder_category", "?")) for r in fn_rows)

    return {
        "train_size": train_size,
        "run_id": run_id,
        "cell_id": cell_id,
        "split": split,
        "ensemble_rule": rule,
        "ensemble_threshold": threshold,
        "models": list(models),
        "csv_path": str(csv_path.resolve()) if csv_path else None,
        "summary_json": str(summary_json.resolve()) if summary_json.is_file() else None,
        "f1_pct": metrics.get("f1_pct"),
        "recall_pct": metrics.get("recall_pct"),
        "fp_rate_pct": metrics.get("fp_rate_pct"),
        "tp": metrics.get("tp"),
        "fp": metrics.get("fp"),
        "fn": metrics.get("fn"),
        "tn": metrics.get("tn"),
        "fp_count": len(fp_rows),
        "fn_count": len(fn_rows),
        "fp_by_category": dict(fp_by_cat.most_common()),
        "fn_by_category": dict(fn_by_cat.most_common()),
        "fp_clips": [
            {
                "folder_category": r.get("folder_category"),
                "clip_name": r.get("clip_name"),
                "p_mean": r.get("p_mean"),
                "clip_path": r.get("clip_path") or r.get("clip_video_path"),
                "uid": r.get("uid"),
            }
            for r in sorted(fp_rows, key=lambda x: -_float(x, "p_mean"))
        ],
        "fn_clips": [
            {
                "folder_category": r.get("folder_category"),
                "clip_name": r.get("clip_name"),
                "p_mean": r.get("p_mean"),
                "clip_path": r.get("clip_path") or r.get("clip_video_path"),
                "uid": r.get("uid"),
            }
            for r in sorted(fn_rows, key=lambda x: _float(x, "p_mean"))
        ],
    }


def analyze_size_run_leaderboard(
    *,
    train_size: int,
    run_id: str,
    cell_id: str,
    split: str,
    task: str,
) -> Dict[str, Any]:
    reports_dir = artifacts_root(run_id) / "reports" / cell_id
    lb_path = reports_dir / f"{split}_leaderboard.csv"
    fp_manifest_path = reports_dir / f"{split}_fp_manifest.csv"

    lb = _read_csv(lb_path)
    best: Dict[str, str] = {}
    if lb:
        prefer = [r for r in lb if str(r.get("decision_mode", "")) == "softmax_argmax"]
        pool = prefer or lb
        best = max(pool, key=lambda r: _float(r, "f1_pct"))

    fp_rows = _read_csv(fp_manifest_path)
    fp_by_cat = Counter(
        str(r.get("folder_category", r.get("category_str", "?"))) for r in fp_rows
    )

    return {
        "train_size": train_size,
        "run_id": run_id,
        "cell_id": cell_id,
        "task": task,
        "split": split,
        "eval_mode": "leaderboard",
        "best_model": best.get("model") or best.get("exp_name"),
        "f1_pct": _float(best, "f1_pct") if best else None,
        "recall_pct": _float(best, "recall_pct") if best else None,
        "fp_rate_pct": _float(best, "fp_rate_pct") if best else None,
        "fp_count": len(fp_rows),
        "fn_count": None,
        "fp_by_category": dict(fp_by_cat.most_common()),
        "fp_clips": [
            {
                "folder_category": r.get("folder_category") or r.get("category_str"),
                "clip_name": r.get("clip_name"),
                "clip_path": r.get("clip_path") or r.get("clip_video_path"),
            }
            for r in fp_rows[:200]
        ],
        "fn_clips": [],
    }


def _load_eval_error_rows(reports_dir: Path, split: str) -> List[Dict[str, str]]:
    """Lee CSV de errores generados por evaluate_campaign (FN/FP con rutas)."""
    candidates = sorted(reports_dir.glob(f"{split}_errors_*.csv"))
    for path in candidates:
        rows = _read_csv(path)
        if rows:
            return rows
    return []


def _enrich_binary_row_from_eval(
    row: Dict[str, Any],
    reports_dir: Path,
    split: str,
    *,
    best_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Si export-errors no corrió, usa best_ensemble.json y errors CSV de eval."""
    if best_spec is None:
        try:
            from evaluate_campaign import load_best_ensemble_spec

            best_spec = load_best_ensemble_spec(reports_dir, split)
        except ImportError:
            best_spec = None

    if best_spec:
        for key in ("f1_pct", "recall_pct", "fp_rate_pct", "fp", "fn", "tp", "tn"):
            if row.get(key) is None and best_spec.get(key) is not None:
                row[key] = best_spec.get(key)
        row["best_ensemble_json"] = str(reports_dir / f"{split}_best_ensemble.json")

    error_rows = _load_eval_error_rows(reports_dir, split)
    if error_rows and not row.get("fp_by_category"):
        fp_rows = [r for r in error_rows if r.get("outcome") == "FP"]
        fn_rows = [r for r in error_rows if r.get("outcome") == "FN"]
        row["fp_count"] = len(fp_rows)
        row["fn_count"] = len(fn_rows)
        row["fp_by_category"] = dict(
            Counter(str(r.get("folder_category", "?")) for r in fp_rows).most_common()
        )
        row["fn_by_category"] = dict(
            Counter(str(r.get("folder_category", "?")) for r in fn_rows).most_common()
        )
        row["fp_clips"] = [
            {
                "folder_category": r.get("folder_category"),
                "clip_name": r.get("clip_name"),
                "clip_path": r.get("clip_path") or r.get("clip_video_path"),
                "uid": r.get("uid"),
            }
            for r in fp_rows[:200]
        ]
        row["fn_clips"] = [
            {
                "folder_category": r.get("folder_category"),
                "clip_name": r.get("clip_name"),
                "clip_path": r.get("clip_path") or r.get("clip_video_path"),
                "uid": r.get("uid"),
            }
            for r in fn_rows[:200]
        ]
        if row.get("fp") is None:
            row["fp"] = len(fp_rows)
        if row.get("fn") is None:
            row["fn"] = len(fn_rows)

    if not row.get("fp_by_category"):
        fp_manifest = reports_dir / f"{split}_fp_manifest.csv"
        fp_rows = _read_csv(fp_manifest)
        if fp_rows:
            row["fp_by_category"] = dict(
                Counter(
                    str(r.get("folder_category", r.get("category_str", "?")))
                    for r in fp_rows
                ).most_common()
            )
            if row.get("fp_count") in (None, 0):
                row["fp_count"] = len(fp_rows)

    return row


def _ensemble_params_from_eval(
    reports_dir: Path,
    split: str,
    config: Dict[str, Any],
    cell_id: str,
    run_id: str,
) -> tuple[str, List[str], float]:
    ens = resolve_ensemble_settings(config, cell_id, run_id=run_id)
    try:
        from evaluate_campaign import load_best_ensemble_spec

        spec = load_best_ensemble_spec(reports_dir, split)
    except ImportError:
        spec = None

    if spec:
        models = [str(m) for m in (spec.get("models") or ens["models"])]
        rule = str(spec.get("rule") or ens["rule"])
        thr = spec.get("threshold")
        if isinstance(thr, list) and thr:
            threshold = float(thr[0])
        elif thr is not None:
            threshold = float(thr)
        else:
            thrs = spec.get("thresholds")
            if isinstance(thrs, str) and thrs.strip():
                threshold = float(str(thrs).split("|")[0])
            elif isinstance(thrs, list) and thrs:
                threshold = float(thrs[0])
            else:
                threshold = float(ens["threshold"])
        return rule, models, threshold

    return str(ens["rule"]), list(ens["models"]), float(ens["threshold"])


def analyze_size_run_for_cell(
    *,
    train_size: int,
    run_id: str,
    cell: Dict[str, Any],
    split: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    cell_id = cell["id"]
    reports_dir = artifacts_root(run_id) / "reports" / cell_id
    if str(cell.get("task")) == "binary":
        rule, models, threshold = _ensemble_params_from_eval(
            reports_dir, split, config, cell_id, run_id
        )
        row = analyze_size_run(
            train_size=train_size,
            run_id=run_id,
            cell_id=cell_id,
            split=split,
            rule=rule,
            models=models,
            threshold=threshold,
        )
        row["task"] = "binary"
        try:
            from evaluate_campaign import load_best_ensemble_spec

            spec = load_best_ensemble_spec(reports_dir, split)
        except ImportError:
            spec = None
        return _enrich_binary_row_from_eval(row, reports_dir, split, best_spec=spec)
    row = analyze_size_run_leaderboard(
        train_size=train_size,
        run_id=run_id,
        cell_id=cell_id,
        split=split,
        task=str(cell.get("task")),
    )
    return row


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})


def summarize_learning_curve(
    *,
    train_sizes: List[int],
    cells: List[Dict[str, Any]],
    config: Dict[str, Any],
    split: Optional[str] = None,
) -> Dict[str, Any]:
    lc = config.get("learning_curve") or {}
    split_name = split or str(lc.get("eval_split") or "val")

    all_size_results: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    fp_cat_rows: List[Dict[str, Any]] = []
    by_cell: Dict[str, Any] = {}

    for cell in cells:
        cid = cell["id"]
        size_results: List[Dict[str, Any]] = []
        for n in train_sizes:
            rid = run_id_for_train_size(n)
            row = analyze_size_run_for_cell(
                train_size=n,
                run_id=rid,
                cell=cell,
                split=split_name,
                config=config,
            )
            size_results.append(row)
            all_size_results.append(row)
            comparison_rows.append({
                "cell_id": cid,
                "task": cell.get("task"),
                "pose_source": cell.get("pose_source"),
                "train_size": n,
                "run_id": rid,
                "f1_pct": row.get("f1_pct"),
                "recall_pct": row.get("recall_pct"),
                "fp_rate_pct": row.get("fp_rate_pct"),
                "fp": row.get("fp") if row.get("fp") is not None else row.get("fp_count"),
                "fn": row.get("fn") if row.get("fn") is not None else row.get("fn_count"),
            })
            for cat, cnt in (row.get("fp_by_category") or {}).items():
                fp_cat_rows.append({
                    "cell_id": cid,
                    "train_size": n,
                    "run_id": rid,
                    "folder_category": cat,
                    "fp_count": cnt,
                })
        by_cell[cid] = {"cell": cell, "sizes": size_results}

    out_dir = master_reports_dir(LEARNING_CURVE_MASTER_RUN_ID) / "learning_curve"
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_csv = out_dir / f"comparison_all_cells_{split_name}.csv"
    _write_csv(
        comparison_csv,
        comparison_rows,
        [
            "cell_id", "task", "pose_source", "train_size", "run_id",
            "f1_pct", "recall_pct", "fp_rate_pct", "fp", "fn",
        ],
    )

    fp_cat_csv = out_dir / f"fp_by_category_all_cells_{split_name}.csv"
    _write_csv(
        fp_cat_csv,
        fp_cat_rows,
        ["cell_id", "train_size", "run_id", "folder_category", "fp_count"],
    )

    fn_list_path = out_dir / f"fn_clips_all_cells_{split_name}.txt"
    with open(fn_list_path, "w", encoding="utf-8") as f:
        for r in all_size_results:
            if str(r.get("task")) != "binary":
                continue
            f.write(f"\n=== {r['cell_id']} train={r['train_size']} FN={r.get('fn_count', '?')} ===\n")
            for clip in r.get("fn_clips") or []:
                f.write(f"  cat={clip.get('folder_category')} clip={clip.get('clip_name')}\n")
                f.write(f"    {clip.get('clip_path')}\n")

    fp_list_path = out_dir / f"fp_clips_all_cells_{split_name}.txt"
    with open(fp_list_path, "w", encoding="utf-8") as f:
        for r in all_size_results:
            f.write(f"\n=== {r['cell_id']} train={r['train_size']} FP={r.get('fp_count', '?')} ===\n")
            f.write(f"  FP por categoría: {r.get('fp_by_category')}\n")
            for clip in (r.get("fp_clips") or [])[:30]:
                f.write(f"  cat={clip.get('folder_category')} clip={clip.get('clip_name')}\n")
                f.write(f"    {clip.get('clip_path')}\n")

    report_json = out_dir / f"learning_curve_report_all_cells_{split_name}.json"
    payload = {
        "cell_ids": [c["id"] for c in cells],
        "split": split_name,
        "train_sizes": train_sizes,
        "comparison": comparison_rows,
        "by_cell": by_cell,
        "outputs": {
            "comparison_csv": str(comparison_csv.resolve()),
            "fp_by_category_csv": str(fp_cat_csv.resolve()),
            "fn_clips_txt": str(fn_list_path.resolve()),
            "fp_clips_txt": str(fp_list_path.resolve()),
            "report_json": str(report_json.resolve()),
        },
    }
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return payload


def print_report(payload: Dict[str, Any]) -> None:
    print(
        f"\n=== Curva de aprendizaje — {len(payload.get('cell_ids') or [])} celdas "
        f"— split {payload.get('split')} ==="
    )
    print(f"{'Celda':<22} | {'Train':>8} | {'F1%':>6} | {'Rec%':>6} | {'FP%':>7} | {'FP':>4} | {'FN':>4}")
    print("-" * 72)
    for row in payload.get("comparison") or []:
        print(
            f"{row.get('cell_id', ''):<22} | {row.get('train_size', ''):>8} | "
            f"{row.get('f1_pct', 0) or 0:>6.1f} | {row.get('recall_pct', 0) or 0:>6.1f} | "
            f"{row.get('fp_rate_pct', 0) or 0:>7.2f} | {row.get('fp', 0) or 0:>4} | "
            f"{row.get('fn', 0) or 0:>4}"
        )

    print("\n--- FP por categoría (top por celda × iteración) ---")
    for cid, block in (payload.get("by_cell") or {}).items():
        for r in block.get("sizes") or []:
            fp_cat = r.get("fp_by_category") or {}
            top = sorted(fp_cat.items(), key=lambda x: -x[1])[:5]
            if top:
                top_str = ", ".join(f"cat{cat}={cnt}" for cat, cnt in top)
            elif r.get("fp_count") == 0 or r.get("fp") == 0:
                top_str = "(0 FP en val)"
            else:
                top_str = "(sin desglose; revisa comparison_csv o reports/)"
            print(f"  {cid} train={r['train_size']}: {top_str}")

    outs = payload.get("outputs") or {}
    print(f"\nInformes:")
    for k, v in outs.items():
        print(f"  {k}: {v}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Comparativa curva de aprendizaje")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--train-sizes", nargs="+", default=None, metavar="N|max")
    ap.add_argument("--split", type=str, default=None, choices=["val", "test"])
    args = ap.parse_args()

    config_path = Path(args.config) if args.config else None
    config = load_merged_campaign_config(config_path)

    try:
        cells = resolve_learning_curve_cells(config, args.cells)
        train_sizes = get_learning_curve_train_sizes(
            cli_sizes=args.train_sizes,
            config=config,
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    payload = summarize_learning_curve(
        train_sizes=train_sizes,
        cells=cells,
        config=config,
        split=args.split,
    )
    print_report(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
