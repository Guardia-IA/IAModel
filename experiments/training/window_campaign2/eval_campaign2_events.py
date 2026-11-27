#!/usr/bin/env python3
"""Event-level OR recall on campaign2 test windows."""
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
from campaign2_metrics import event_recall_or  # noqa: E402
from campaign2_paths import load_campaign2_config, models_dir  # noqa: E402
from campaign2_scoring import load_model_from_checkpoint  # noqa: E402


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--exp-id", type=int, default=49)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    cfg = load_campaign2_config()
    cell = str(cfg.get("default_cell_id"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = models_dir(cell) / f"modelo_{args.exp_id:02d}.pt"
    if not path.is_file():
        raise SystemExit(f"Missing {path}")

    loaders, *_rest, summary = build_campaign2_datasets_and_loaders(
        data_root=Path(args.data_root).resolve(),
        cfg=cfg,
        batch_size=32,
        num_workers=0,
        augment_on_the_fly=False,
        cell_id=cell,
    )
    model, _ = load_model_from_checkpoint(path, device, kind="window_native")
    probs: list[float] = []
    uids: list[str] = []
    flags: list[bool] = []
    test_ds = loaders["test"].dataset
    for batch in loaders["test"]:
        x = batch[0].to(device)
        logits = model(x)
        p = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs.extend(p.tolist())

    for w in test_ds.windows:
        uids.append(w.clip_uid)
        flags.append(int(w.train_label) == 1)

    ev = event_recall_or(
        clip_uids=uids,
        clip_is_positive=flags,
        window_probs=np.asarray(probs),
        window_clip_uid=uids,
        threshold=args.threshold,
    )
    print(f"[EVENT] exp {args.exp_id:02d} @ {args.threshold}: {ev}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())