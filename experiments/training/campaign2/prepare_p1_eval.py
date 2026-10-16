#!/usr/bin/env python3
"""Genera cola jobs/queue.json solo para eval P1."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CAMPAIGN2_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN2_DIR.parent
for p in (TRAINING_DIR, CAMPAIGN2_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from campaign2_paths import load_campaign2_config, read_current_run_id, run_layout  # noqa: E402
from p1_eval_jobs import build_p1_eval_job_queue, write_eval_queue  # noqa: E402


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
    manifest = layout["cache_dir"] / "manifest.json"
    if not manifest.is_file():
        print(f"ERROR: falta {manifest}", file=sys.stderr)
        return 2

    jobs, warnings = build_p1_eval_job_queue(
        cfg,
        manifest_path=str(manifest),
        run_id=run_id,
        jobs_dir=layout["jobs_dir"],
    )
    for w in warnings:
        print(f"WARN: {w}")
    if not jobs:
        return 2
    qpath = write_eval_queue(run_id, jobs)
    print(f"Cola eval P1: {len(jobs)} jobs → {qpath}")
    for j in jobs:
        print(f"  - {j.job_id} ← {j.params.get('source_job_id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
