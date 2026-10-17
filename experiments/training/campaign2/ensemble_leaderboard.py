#!/usr/bin/env python3
"""
Ranking de ensembles (AND @ umbral) sobre predicciones P0/P1 por pose_source.

Busca el mejor par entre modelos legacy evaluados y/o checkpoints v2.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
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

BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
RESET = "\033[0m"


def _load_predictions(job_dir: Path) -> Optional[Dict[str, Any]]:
    for name in ("clip_predictions.json",):
        p = job_dir / name
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                preds = json.load(f)
            meta_path = job_dir / "p0_result.json"
            if not meta_path.is_file():
                meta_path = job_dir / "p1_eval_result.json"
            meta: Dict[str, Any] = {}
            if meta_path.is_file():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            return {"preds": preds, "meta": meta, "job_dir": str(job_dir)}
    return None


def _infer_model_key(job_name: str) -> Tuple[str, str, str]:
    """Returns (model_id, pose_source, kind)."""
    ps = "full"
    for tag in ("full", "filtered"):
        if f"_{tag}_" in job_name or job_name.endswith(f"_{tag}"):
            ps = tag
            break
        if f"_{tag}" in job_name:
            ps = tag
    m = re.search(r"_m(\d+)", job_name)
    mid = m.group(1) if m else job_name
    if job_name.startswith("eval_"):
        inner = job_name.replace("eval_", "", 1)
        if inner.startswith("p0"):
            pass
        else:
            mid = inner.split("_s")[0].replace("_full", "").replace("_filtered", "")
    if job_name.startswith("p0"):
        kind = "legacy"
    elif job_name.startswith("eval_v2") or "v2_" in job_name or "dbg_" in job_name:
        kind = "v2"
    else:
        kind = "eval"
    return mid, ps, kind


def _by_uid(preds: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r["uid"]): r for r in preds}


def _ensemble_and(
    pa: Dict[str, Dict[str, Any]],
    pb: Dict[str, Dict[str, Any]],
    thr_a: float,
    thr_b: float,
) -> Dict[str, Any]:
    uids = sorted(set(pa) & set(pb))
    yt, pred = [], []
    for u in uids:
        yt.append(int(pa[u]["true_label"]))
        a = float(pa[u]["prob_pos"]) >= thr_a
        b = float(pb[u]["prob_pos"]) >= thr_b
        pred.append(1 if (a and b) else 0)
    yt_arr = np.array(yt, dtype=np.int64)
    pr_arr = np.array(pred, dtype=np.int64)
    return _binary_metrics(yt_arr, pr_arr)


def collect_scored_jobs(jobs_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for d in sorted(jobs_dir.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("eval_"):
            src = d.name.replace("eval_", "", 1)
            # eval of v2 points to train job predictions file sometimes in sibling
            pass
        bundle = _load_predictions(d)
        if not bundle:
            continue
        name = d.name
        if name.startswith("eval_"):
            name = name.replace("eval_", "", 1)
        mid, ps, kind = _infer_model_key(name)
        rows.append(
            {
                "job_id": d.name,
                "model_key": mid,
                "pose_source": ps,
                "kind": kind,
                "preds_by_uid": _by_uid(bundle["preds"]),
                "meta": bundle["meta"],
            }
        )
    # Also train jobs with only p1_eval sibling — prefer eval_* dirs
    eval_names = {r["job_id"] for r in rows}
    for d in sorted(jobs_dir.iterdir()):
        if not d.is_dir() or d.name in eval_names:
            continue
        if (d / "clip_predictions.json").is_file() and d.name.startswith("p0"):
            bundle = _load_predictions(d)
            if bundle:
                mid, ps, kind = _infer_model_key(d.name)
                rows.append(
                    {
                        "job_id": d.name,
                        "model_key": mid,
                        "pose_source": ps,
                        "kind": kind,
                        "preds_by_uid": _by_uid(bundle["preds"]),
                        "meta": bundle["meta"],
                    }
                )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()

    run_id = args.run_id or read_current_run_id()
    if not run_id:
        print("ERROR: sin RUN_ID", file=sys.stderr)
        return 2
    cfg = load_campaign2_config(Path(args.config) if args.config else None)
    layout = run_layout(run_id)
    mx = cfg.get("campaign_matrix") or {}
    thresholds = (cfg.get("p0_incumbent") or {}).get("frozen_thresholds") or {"41": 0.8, "48": 0.8}
    default_thr = 0.8
    pairs_cfg = list(mx.get("ensemble_pairs") or [])

    scored = collect_scored_jobs(layout["jobs_dir"])
    if not scored:
        print("Sin predicciones clip-level (ejecuta p0/p1-eval antes).")
        return 1

    # Index: pose_source -> model_key -> best job (prefer p0b windows)
    index: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in scored:
        ps = row["pose_source"]
        mk = str(row["model_key"])
        index.setdefault(ps, {})
        prev = index[ps].get(mk)
        if prev is None or "p0b" in row["job_id"]:
            index[ps][mk] = row

    leaderboard: List[Dict[str, Any]] = []
    for ps, models in index.items():
        keys = sorted(models.keys(), key=lambda x: (len(x), x))
        pair_list: List[Tuple[str, str]] = []
        for p in pairs_cfg:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                pair_list.append((str(p[0]), str(p[1])))
        if not pair_list and len(keys) >= 2:
            # todas las parejas de modelos legacy numéricos presentes
            legacy_keys = [k for k in keys if k.isdigit()]
            pair_list = list(itertools.combinations(sorted(legacy_keys, key=int), 2))
        for a, b in pair_list:
            if a not in models or b not in models:
                continue
            ta = float(thresholds.get(str(a), default_thr))
            tb = float(thresholds.get(str(b), default_thr))
            metrics = _ensemble_and(models[a]["preds_by_uid"], models[b]["preds_by_uid"], ta, tb)
            score = float(metrics["recall_pct"]) - 0.5 * float(metrics["fp_rate_pct"])
            leaderboard.append(
                {
                    "pose_source": ps,
                    "model_a": a,
                    "model_b": b,
                    "threshold_a": ta,
                    "threshold_b": tb,
                    "rule": "and",
                    "job_a": models[a]["job_id"],
                    "job_b": models[b]["job_id"],
                    "metrics": metrics,
                    "score": score,
                }
            )
        for mk, row in models.items():
            uids = row["preds_by_uid"]
            yt = np.array([int(uids[u]["true_label"]) for u in uids], dtype=np.int64)
            pr = np.array([float(uids[u]["prob_pos"]) for u in uids], dtype=np.float64)
            thr = float(thresholds.get(str(mk), default_thr)) if mk.isdigit() else default_thr
            pred = (pr >= thr).astype(np.int64)
            sm = _binary_metrics(yt, pred)
            leaderboard.append(
                {
                    "pose_source": ps,
                    "model_a": mk,
                    "model_b": None,
                    "rule": "single",
                    "job_a": row["job_id"],
                    "metrics": sm,
                    "score": float(sm["recall_pct"]) - 0.5 * float(sm["fp_rate_pct"]),
                }
            )

    leaderboard.sort(key=lambda x: -float(x.get("score", 0)))
    report = {"run_id": run_id, "leaderboard": leaderboard}
    out = layout["reports_dir"] / "ensemble_leaderboard.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"{BOLD}{'═' * 72}{RESET}")
    print(f"{BOLD}Ensemble leaderboard{RESET} — {run_id}")
    print(f"{BOLD}{'═' * 72}{RESET}")
    for i, row in enumerate(leaderboard[:15]):
        m = row["metrics"]
        if row.get("model_b"):
            label = f"{row['pose_source']} AND #{row['model_a']} + #{row['model_b']}"
        else:
            label = f"{row['pose_source']} single #{row['model_a']}"
        star = f"{GREEN}★{RESET} " if i == 0 else "  "
        print(
            f"{star}{label}: Rec={m.get('recall_pct', 0):.1f}% "
            f"FP={m.get('fp')} F1={m.get('f1_pct', 0):.1f}% score={row.get('score', 0):.2f}"
        )
    print(f"\nJSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
