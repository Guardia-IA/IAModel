#!/usr/bin/env python3
"""
Señal rápida (~3–8 min): pocos clips, sin quality, 1 época, mira rec_robo en val.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

WINDOW_DIR = Path(__file__).resolve().parent
TRAINING_DIR = WINDOW_DIR.parent
for p in (TRAINING_DIR, WINDOW_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch

from train_window_campaign import run_window_experiment
from window_paths import load_window_config, models_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Quick train signal (few clips, 1 epoch)")
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--experiment-id", type=int, default=48)
    ap.add_argument("--clips", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()

    cfg = dict(load_window_config())
    cfg["_debug_max_clips"] = int(args.clips)
    td = dict(cfg.get("train_defaults") or {})
    td["epochs"] = int(args.epochs)
    cfg["train_defaults"] = td
    qg = dict(cfg.get("quality_gate") or {})
    qg["enabled"] = False
    cfg["quality_gate"] = qg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cell_id = str(cfg.get("default_cell_id", "win_bin_full_baseline"))
    out = models_dir(cell_id) / "_quick_signal"
    out.mkdir(parents=True, exist_ok=True)

    print(
        f"[QUICK] clips={args.clips} epochs={args.epochs} sin quality_gate | device={device}",
        flush=True,
    )
    run_window_experiment(
        int(args.experiment_id),
        data_root=Path(args.data_root).resolve(),
        device=device,
        cfg_window=cfg,
        task="binary",
        positive_classes=cfg.get("robbery_classes", [6, 14]),
        augment_on_the_fly=False,
        manifest_cache_dir=None,
        models_out=out,
        progress_path=None,
    )
    print("[QUICK] Si rec_robo > 0 en la línea [Win ..] ep 001 → pipeline OK.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())