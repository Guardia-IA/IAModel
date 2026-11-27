#!/usr/bin/env python3
"""Train window_campaign2 experiments (native 8s, v2 labels)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent
_TR = _ROOT.parent
_WC = _TR / "window_campaign"
for p in (_TR, _WC, _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from campaign2_paths import load_campaign2_config, models_dir  # noqa: E402
from train_campaign2_core import run_campaign2_experiment  # noqa: E402
from train_model_operations import get_data_result_roots  # noqa: E402
from window_train_progress import DEFAULT_PROGRESS_PATH, attach_heartbeat, finish_experiment, init_batch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="window_campaign2 training")
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--experiment-id", type=int, nargs="+", default=None)
    ap.add_argument(
        "--stage",
        choices=("pilot", "benchmark", "optional_48"),
        default="benchmark",
    )
    ap.add_argument("--cell-id", type=str, default=None)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--force-new-split", action="store_true")
    ap.add_argument("--debug-clips", type=int, default=0)
    ap.add_argument("--manifest-cache-dir", type=str, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_campaign2_config()
    if args.force_new_split:
        cfg["_force_new_split"] = True
    if args.debug_clips > 0:
        cfg["_debug_max_clips"] = args.debug_clips
    cell = args.cell_id or cfg.get("default_cell_id")
    stages = cfg.get("experiment_stages") or {}
    exp_ids = args.experiment_id or list(stages.get(args.stage) or cfg.get("experiment_benchmark", [49, 53, 11]))

    data_root = Path(args.data_root).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = models_dir(str(cell))
    mdir = Path(args.manifest_cache_dir).resolve() if args.manifest_cache_dir else None
    augment = not args.no_augment and bool((cfg.get("synthetic_variants") or {}).get("augment_on_the_fly", True))

    progress_path = _ROOT / "artifacts" / "logs" / "train_progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    init_batch(
        path=progress_path,
        stage=args.stage,
        exp_ids=exp_ids,
        parallel=1,
        cell_id=str(cell),
        data_root=str(data_root),
    )
    hb = attach_heartbeat(path=progress_path, interval_sec=60.0)
    try:
        for eid in exp_ids:
            ckpt = out / f"modelo_{eid:02d}.pt"
            if args.resume and ckpt.is_file():
                print(f"[WIN2] Skip {eid:02d} (exists)", flush=True)
                finish_experiment(progress_path, eid, ok=True, wall_seconds=0.0)
                continue
            try:
                run_campaign2_experiment(
                    eid,
                    data_root=data_root,
                    device=device,
                    cfg=cfg,
                    augment_on_the_fly=augment,
                    manifest_cache_dir=mdir,
                    models_out=out,
                    progress_path=progress_path,
                )
                finish_experiment(progress_path, eid, ok=True)
            except Exception as exc:
                finish_experiment(progress_path, eid, ok=False, error=str(exc))
                raise
    finally:
        hb.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())