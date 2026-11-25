#!/usr/bin/env python3
"""
Flujo operativo: preparar datos → entrenar N experimentos en serie → resumen CSV.

  python3 run_overnight_batch.py --data-root /path/data_result --stage benchmark
  nohup python3 run_overnight_batch.py ... > artifacts/logs/overnight.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

WINDOW_DIR = Path(__file__).resolve().parent


def _run(cmd: list[str], cwd: Path) -> int:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(cwd))


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch overnight window campaign")
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument(
        "--stage",
        choices=("pilot", "stage2", "stage3", "benchmark", "full"),
        default="benchmark",
    )
    ap.add_argument("--cell-id", type=str, default=None)
    ap.add_argument("--skip-core-index", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--manifest-cache-dir", type=str, default=None)
    ap.add_argument("--debug-clips", type=int, default=0)
    ap.add_argument("--debug-epochs", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="Reenvía a train_window_campaign (default: window_config parallelism)",
    )
    ap.add_argument("--resume", action="store_true", help="Omitir modelos ya entrenados")
    args = ap.parse_args()

    py = sys.executable
    data_root = str(Path(args.data_root).resolve())
    log_dir = WINDOW_DIR / "artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    manifest_path = log_dir / f"batch_manifest_{stamp}.json"

    stages_order = ["benchmark"] if args.stage == "full" else [args.stage]

    preflight = [py, "preflight_window_campaign.py", "--data-root", data_root, "--write-plan"]
    if args.debug_clips > 0:
        preflight.extend(["--debug-clips", str(args.debug_clips)])

    steps: list[list[str]] = []
    if not args.skip_core_index:
        steps.append([py, "build_core_index.py", "--data-root", data_root])
    if not args.skip_preflight:
        steps.append(preflight)

    for st in stages_order:
        cmd = [py, "train_window_campaign.py", "--data-root", data_root, "--stage", st]
        if args.cell_id:
            cmd.extend(["--cell-id", args.cell_id])
        if args.manifest_cache_dir:
            cmd.extend(["--manifest-cache-dir", args.manifest_cache_dir])
        if args.debug_clips > 0:
            cmd.extend(["--debug-clips", str(args.debug_clips)])
        if args.debug_epochs > 0:
            cmd.extend(["--debug-epochs", str(args.debug_epochs)])
        if args.parallel is not None:
            cmd.extend(["--parallel", str(args.parallel)])
        if args.resume:
            cmd.append("--resume")
        steps.append(cmd)

    steps.append([py, "summarize_window_runs.py"] + (["--cell-id", args.cell_id] if args.cell_id else []))
    steps.append(
        [py, "verify_split_disjoint.py", "--data-root", data_root]
        + (["--cell-id", args.cell_id] if args.cell_id else [])
    )

    t0 = time.perf_counter()
    results: list[dict] = []
    rc = 0
    for cmd in steps:
        if args.dry_run:
            print("DRY:", cmd)
            results.append({"cmd": cmd, "exit": 0})
            continue
        code = _run(cmd, WINDOW_DIR)
        results.append({"cmd": cmd, "exit": code})
        if code != 0:
            rc = code
            print(f"[BATCH] Step failed exit={code}; continuing with next steps.")

    elapsed = time.perf_counter() - t0
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "started": stamp,
                "data_root": data_root,
                "stages": stages_order,
                "elapsed_seconds": elapsed,
                "steps": results,
            },
            f,
            indent=2,
        )
    print(f"\n[BATCH] Done in {elapsed/3600:.2f} h | manifest {manifest_path} | last_rc={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())