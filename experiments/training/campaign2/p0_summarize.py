#!/usr/bin/env python3
"""Agrega resultados P0 y calcula ensemble AND @ umbral congelado."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

CAMPAIGN2_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN2_DIR.parent
CAMPAIGN_DIR = TRAINING_DIR / "campaign"
for p in (TRAINING_DIR, CAMPAIGN2_DIR, CAMPAIGN_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from campaign2_paths import load_campaign2_config, read_current_run_id, run_layout  # noqa: E402
from evaluate_campaign import _binary_metrics  # noqa: E402
from p0_display import render_p0_baseline  # noqa: E402


def _load_job_p0_result(jobs_dir: Path, job_id: str) -> Optional[Dict[str, Any]]:
    p = jobs_dir / job_id / "p0_result.json"
    if not p.is_file():
        r = jobs_dir / job_id / "result.json"
        if r.is_file():
            with open(r, "r", encoding="utf-8") as f:
                wrap = json.load(f)
            if wrap.get("status") != "ok":
                return None
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_predictions(jobs_dir: Path, job_id: str) -> List[Dict[str, Any]]:
    pred = jobs_dir / job_id / "clip_predictions.json"
    if not pred.is_file():
        return []
    with open(pred, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensemble_and(
    preds_a: List[Dict[str, Any]],
    preds_b: List[Dict[str, Any]],
    thr_a: float,
    thr_b: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_a = {str(r["uid"]): r for r in preds_a}
    by_b = {str(r["uid"]): r for r in preds_b}
    uids = sorted(set(by_a) & set(by_b))
    y_true: List[int] = []
    probs_a: List[float] = []
    probs_b: List[float] = []
    for u in uids:
        ra, rb = by_a[u], by_b[u]
        y_true.append(int(ra["true_label"]))
        probs_a.append(float(ra["prob_pos"]))
        probs_b.append(float(rb["prob_pos"]))
    yt = np.array(y_true, dtype=np.int64)
    pa = np.array(probs_a, dtype=np.float64)
    pb = np.array(probs_b, dtype=np.float64)
    pred = ((pa >= thr_a) & (pb >= thr_b)).astype(np.int64)
    return yt, pred, pa


def summarize_run(run_id: str, config_path: Optional[Path] = None) -> Dict[str, Any]:
    cfg = load_campaign2_config(config_path)
    layout = run_layout(run_id)
    jobs_dir = layout["jobs_dir"]
    p0_cfg = cfg.get("p0_incumbent") or {}
    thresholds = p0_cfg.get("frozen_thresholds") or {"41": 0.8, "48": 0.8}

    per_job: List[Dict[str, Any]] = []
    if jobs_dir.is_dir():
        for sub in sorted(jobs_dir.iterdir()):
            if not sub.is_dir():
                continue
            res = _load_job_p0_result(jobs_dir, sub.name)
            if res:
                per_job.append({"job_id": sub.name, **res})

    ensembles: List[Dict[str, Any]] = []
    by_key: Dict[Tuple[str, bool], Dict[str, str]] = {}
    for row in per_job:
        mode = str(row.get("mode"))
        su = bool(row.get("single_user_only"))
        mid = str(row.get("model_id"))
        by_key.setdefault((mode, su), {})[mid] = row["job_id"]

    for (mode, su), mids in sorted(by_key.items()):
        if "41" in mids and "48" in mids:
            ja, jb = mids["41"], mids["48"]
            pa = _load_predictions(jobs_dir, ja)
            pb = _load_predictions(jobs_dir, jb)
            thr_a = float(thresholds.get("41", 0.8))
            thr_b = float(thresholds.get("48", 0.8))
            yt, pred, _pa = _ensemble_and(pa, pb, thr_a, thr_b)
            metrics = _binary_metrics(yt, pred)
            ensembles.append(
                {
                    "mode": mode,
                    "single_user_only": su,
                    "rule": "and",
                    "thresholds": {"41": thr_a, "48": thr_b},
                    "jobs": {"41": ja, "48": jb},
                    "metrics": metrics,
                }
            )

    report = {
        "run_id": run_id,
        "phase": "sprint2_p0",
        "eval_split": p0_cfg.get("eval_split", "val_dev"),
        "frozen_thresholds": thresholds,
        "ensemble_rule": p0_cfg.get("ensemble_rule", "and"),
        "per_model_jobs": per_job,
        "ensembles": ensembles,
        "num_jobs_ok": len(per_job),
    }
    out_path = layout["reports_dir"] / "p0_baseline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    report["_json_path"] = str(out_path)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Resumen P0 → reports/p0_baseline.json")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--run-id", type=str, default=None)
    args = ap.parse_args()

    run_id = args.run_id or read_current_run_id()
    if not run_id:
        print("ERROR: sin RUN_ID", file=sys.stderr)
        return 2

    report = summarize_run(run_id, Path(args.config) if args.config else None)
    print(render_p0_baseline(report))
    print(f"\n(JSON: {report['_json_path']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
