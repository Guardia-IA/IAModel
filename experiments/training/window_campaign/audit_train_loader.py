#!/usr/bin/env python3
"""Audita etiquetas y proporción pos/neg en el train loader (sin entrenar)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

WINDOW_DIR = Path(__file__).resolve().parent
TRAINING_DIR = WINDOW_DIR.parent
for p in (TRAINING_DIR, WINDOW_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from window_build_loaders import build_window_datasets_and_loaders
from window_paths import load_window_config


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit window train loader labels")
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument(
        "--debug-clips",
        type=int,
        default=80,
        help="Solo N clips (rápido). 0 = corpus completo (lento).",
    )
    ap.add_argument("--max-batches", type=int, default=20)
    ap.add_argument(
        "--skip-quality",
        action="store_true",
        help="Omite quality_gate (solo audit; ahorra muchos minutos).",
    )
    args = ap.parse_args()

    cfg = load_window_config()
    cfg = dict(cfg)
    if args.debug_clips > 0:
        cfg["_debug_max_clips"] = int(args.debug_clips)
    if args.skip_quality:
        qg = dict(cfg.get("quality_gate") or {})
        qg["enabled"] = False
        cfg["quality_gate"] = qg
        print("[AUDIT] quality_gate desactivado (solo esta corrida)", flush=True)

    data_root = Path(args.data_root).resolve()
    batch_size = int((cfg.get("train_defaults") or {}).get("batch_size", 32))
    loaders, _dim, label_to_idx, summary = build_window_datasets_and_loaders(
        data_root=data_root,
        cfg_window=cfg,
        batch_size=batch_size,
        num_workers=0,
        augment_on_the_fly=False,
        force_new_split=False,
        cell_id=cfg.get("default_cell_id"),
    )
    train = loaders["train"]
    sampler = getattr(train, "sampler", None)
    if sampler is not None and hasattr(sampler, "stats_line"):
        print(f"[AUDIT] {sampler.stats_line()}")
    print(f"[AUDIT] label_to_idx={label_to_idx}")
    print(f"[AUDIT] label_states train={summary.get('label_states', {}).get('train')}")

    n_y0 = n_y1 = n_wpos = 0
    n_batches = 0
    for bi, batch in enumerate(train):
        if bi >= int(args.max_batches):
            break
        n_batches += 1
        if len(batch) == 3:
            _x, y, w = batch
        else:
            _x, y = batch
            w = torch.ones_like(y, dtype=torch.float32)
        n_y0 += int((y == 0).sum())
        n_y1 += int((y == 1).sum())
        n_wpos += int((w > 1.01).sum())

    tot = max(1, n_y0 + n_y1)
    print(
        f"[AUDIT] batches={n_batches} "
        f"| y=0:{n_y0} y=1:{n_y1} | frac_pos={100.0 * n_y1 / tot:.1f}% "
        f"| sample_weight>1:{n_wpos}"
    )
    if n_y1 == 0:
        print("[AUDIT] FAIL: ningún y=1 en train loader — explica rec_robo=0.")
        return 1
    print("[AUDIT] OK: hay muestras positivas en train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())