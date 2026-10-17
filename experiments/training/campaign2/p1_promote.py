#!/usr/bin/env python3
"""Promueve el mejor checkpoint P1 a models/promoted/ (handoff C2-lite)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

CAMPAIGN2_DIR = Path(__file__).resolve().parent
for p in (CAMPAIGN2_DIR.parent, CAMPAIGN2_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from campaign2_paths import read_current_run_id, run_layout  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    args = ap.parse_args()
    run_id = args.run_id or read_current_run_id()
    if not run_id:
        print("ERROR: sin RUN_ID", file=sys.stderr)
        return 2

    layout = run_layout(run_id)
    summary_path = layout["reports_dir"] / "p1_train_summary.json"
    if not summary_path.is_file():
        print("ERROR: ejecuta p1-summary antes de promote", file=sys.stderr)
        return 2
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    jobs = summary.get("jobs") or []
    if not jobs:
        print("ERROR: sin jobs en p1_train_summary.json", file=sys.stderr)
        return 2
    best = max(jobs, key=lambda x: float(x.get("best_score") or -1e9))
    ckpt = Path(str(best.get("checkpoint") or ""))
    if not ckpt.is_file():
        print(f"ERROR: falta checkpoint {ckpt}", file=sys.stderr)
        return 2

    promoted_dir = CAMPAIGN2_DIR / "models" / "promoted"
    promoted_dir.mkdir(parents=True, exist_ok=True)
    dest_ckpt = promoted_dir / f"{run_id}_{best['job_id']}_best.pt"
    shutil.copy2(ckpt, dest_ckpt)
    thr_src = ckpt.parent / "threshold.json"
    thr_doc = None
    if thr_src.is_file():
        thr_doc = json.loads(thr_src.read_text(encoding="utf-8"))
        shutil.copy2(thr_src, promoted_dir / f"{run_id}_{best['job_id']}_threshold.json")

    handoff = {
        "run_id": run_id,
        "job_id": best["job_id"],
        "checkpoint": str(dest_ckpt),
        "threshold": thr_doc,
        "metrics": {
            "best_score": best.get("best_score"),
            "val_recall_pct": best.get("val_recall_pct"),
            "val_fp": best.get("val_fp"),
        },
    }
    handoff_path = promoted_dir / f"{run_id}_handoff.json"
    with open(handoff_path, "w", encoding="utf-8") as f:
        json.dump(handoff, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Promovido: {dest_ckpt}")
    print(f"Handoff: {handoff_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
