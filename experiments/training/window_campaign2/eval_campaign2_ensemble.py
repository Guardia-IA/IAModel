#!/usr/bin/env python3
"""Ensemble eval on campaign2 test loader."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
TR = ROOT.parent
WC = TR / "window_campaign"
for p in (TR, WC, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from campaign2_build_loaders import build_campaign2_datasets_and_loaders  # noqa: E402
from campaign2_paths import load_campaign2_config, models_dir  # noqa: E402
from campaign2_scoring import load_model_from_checkpoint  # noqa: E402
from model_config import EXPERIMENTS  # type: ignore


def _metrics(y: np.ndarray, pred: np.ndarray) -> str:
    pos = y == 1
    rec = 100.0 * (pred & pos).sum() / max(int(pos.sum()), 1)
    fpr = 100.0 * (pred & ~pos).sum() / max(int((~pos).sum()), 1)
    return f"rec_robo={rec:.1f}% FP={fpr:.1f}% n_pos={int(pos.sum())}"


@torch.no_grad()
def _probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    ps, ys = [], []
    model.eval()
    for batch in loader:
        x, y = batch[0], batch[1]
        logits = model(x.to(device))
        p = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        ps.append(p)
        ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--pairs", nargs="+", default=None)
    args = ap.parse_args()

    cfg = load_campaign2_config()
    cell = str(cfg.get("default_cell_id"))
    ev = cfg.get("evaluation") or {}
    pairs = args.pairs or ev.get("ensemble_pairs", ["49,53", "11,49"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdir = models_dir(cell)

    loaders, *_rest, summary = build_campaign2_datasets_and_loaders(
        data_root=Path(args.data_root).resolve(),
        cfg=cfg,
        batch_size=32,
        num_workers=0,
        augment_on_the_fly=False,
        cell_id=cell,
    )
    print(f"[EVAL2] test label_states={summary['label_states']['test']}", flush=True)

    needed = sorted({int(x) for p in pairs for x in p.split(",")})
    probs: dict[int, np.ndarray] = {}
    y_ref = None
    for eid in needed:
        path = mdir / f"modelo_{eid:02d}.pt"
        if not path.is_file():
            print(f"[EVAL2] skip missing {path}")
            continue
        model, _ = load_model_from_checkpoint(path, device, kind="window_native")
        p, y = _probs(model, loaders["test"], device)
        probs[eid] = p
        y_ref = y
        del model

    if y_ref is None:
        raise SystemExit("No checkpoints scored")
    for eid, p in probs.items():
        print(f"  solo {eid:02d} @0.50  {_metrics(y_ref, p >= 0.50)}")
        print(f"  solo {eid:02d} @0.80  {_metrics(y_ref, p >= 0.80)}")
    for pair in pairs:
        a, b = (int(x) for x in pair.split(","))
        if a not in probs or b not in probs:
            continue
        pa, pb = probs[a], probs[b]
        mean = (pa + pb) / 2
        print(f"\n  par {a:02d}+{b:02d}")
        print(f"    mean @0.50  {_metrics(y_ref, mean >= 0.50)}")
        print(f"    AND  @0.50  {_metrics(y_ref, (pa >= 0.50) & (pb >= 0.50))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())