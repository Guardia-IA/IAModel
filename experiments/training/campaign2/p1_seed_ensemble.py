#!/usr/bin/env python3
"""
Ensemble v2: media de prob clip-max entre semillas del mismo job base (experto: media márgenes).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

CAMPAIGN2_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN2_DIR.parent
CAMPAIGN_DIR = TRAINING_DIR / "campaign"
for p in (TRAINING_DIR, CAMPAIGN2_DIR, CAMPAIGN_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from campaign2_paths import load_campaign2_config, read_current_run_id, run_layout  # noqa: E402
from evaluate_campaign import _binary_metrics  # noqa: E402

_SEED_SUFFIX = re.compile(r"_s(\d+)$")


def _base_job_id(job_name: str) -> str:
    return _SEED_SUFFIX.sub("", job_name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--threshold", type=float, default=None, help="Si no se indica, usa threshold.json del primer seed")
    args = ap.parse_args()

    run_id = args.run_id or read_current_run_id()
    if not run_id:
        print("ERROR: sin RUN_ID", file=sys.stderr)
        return 2
    layout = run_layout(run_id)
    jobs_dir = layout["jobs_dir"]

    groups: Dict[str, List[Path]] = defaultdict(list)
    for d in sorted(jobs_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("eval_"):
            continue
        if not (d / "clip_predictions.json").is_file():
            continue
        src = d.name.replace("eval_", "", 1)
        groups[_base_job_id(src)].append(d)

    rows: List[Dict[str, Any]] = []
    for base, dirs in sorted(groups.items()):
        if len(dirs) < 2:
            continue
        by_uid: Dict[str, List[float]] = defaultdict(list)
        y_by_uid: Dict[str, int] = {}
        thr = args.threshold
        for d in dirs:
            if thr is None:
                src_job = d.name.replace("eval_", "", 1)
                tpath = jobs_dir / src_job / "threshold.json"
                if tpath.is_file():
                    with open(tpath, "r", encoding="utf-8") as f:
                        thr = float(json.load(f).get("threshold", 0.5))
            with open(d / "clip_predictions.json", "r", encoding="utf-8") as f:
                preds = json.load(f)
            for p in preds:
                u = str(p["uid"])
                by_uid[u].append(float(p["prob_pos"]))
                y_by_uid[u] = int(p["true_label"])
        if thr is None:
            thr = 0.5
        uids = sorted(by_uid.keys())
        yt = np.array([y_by_uid[u] for u in uids], dtype=np.int64)
        pr = np.array([float(np.mean(by_uid[u])) for u in uids], dtype=np.float64)
        pred = (pr >= thr).astype(np.int64)
        m = _binary_metrics(yt, pred)
        rows.append(
            {
                "base_job_id": base,
                "n_seeds": len(dirs),
                "threshold": float(thr),
                "rule": "mean_prob",
                "eval_jobs": [d.name for d in dirs],
                "metrics": m,
                "score": float(m["recall_pct"]) - 0.5 * float(m["fp_rate_pct"]),
            }
        )

    rows.sort(key=lambda r: -float(r.get("score", 0)))
    out = layout["reports_dir"] / "p1_seed_ensemble.json"
    report = {"run_id": run_id, "ensembles": rows}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Ensembles v2 (media semillas): {len(rows)}")
    for r in rows[:10]:
        m = r["metrics"]
        print(
            f"  {r['base_job_id']} ({r['n_seeds']} seeds): "
            f"Rec={m.get('recall_pct', 0):.1f}% FP={m.get('fp')} thr={r['threshold']:.3f}"
        )
    print(f"JSON: {out}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
