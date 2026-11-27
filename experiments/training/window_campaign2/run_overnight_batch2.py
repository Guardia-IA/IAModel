#!/usr/bin/env python3
"""Overnight batch: core index → optional phase0 → train benchmark."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str]) -> int:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--stage", default="benchmark", choices=("pilot", "benchmark", "optional_48"))
    ap.add_argument("--skip-phase0", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--force-new-split", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--full-clip-dir", type=str, default=None, help="campaign bin_full models for P0")
    args = ap.parse_args()

    py = sys.executable
    data = str(Path(args.data_root).resolve())
    log_dir = ROOT / "artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if not args.skip_phase0:
        p0 = [py, "phase0/run_phase0.py", "--data-root", data, "--split", "test"]
        if args.full_clip_dir:
            p0.extend(["--full-clip-checkpoint-dir", args.full_clip_dir])
        rc = _run(p0)
        if rc != 0:
            return rc

    if not args.skip_train:
        cmd = [py, "train_campaign2.py", "--data-root", data, "--stage", args.stage]
        if args.force_new_split:
            cmd.append("--force-new-split")
        if args.resume:
            cmd.append("--resume")
        rc = _run(cmd)
        if rc != 0:
            return rc
        _run([py, "eval_campaign2_ensemble.py", "--data-root", data])
        _run([py, "eval_campaign2_events.py", "--data-root", data])

    print(f"[BATCH2] done stamp={stamp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())