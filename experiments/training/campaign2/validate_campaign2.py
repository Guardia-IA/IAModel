#!/usr/bin/env python3
"""Comprobaciones rápidas sin GPU ni datos (≈5 s)."""
from __future__ import annotations

import sys
from pathlib import Path

CAMPAIGN2_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN2_DIR.parent
for p in (TRAINING_DIR, CAMPAIGN2_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _check_metrics() -> None:
    import numpy as np

    from train_v2.metrics import (
        clip_metrics_at_fp_budget,
        threshold_from_negative_order_statistic,
    )

    yt = np.array([1, 0, 0, 0, 1, 0], dtype=np.int64)
    pr = np.array([0.9, 0.7, 0.3, 0.2, 0.8, 0.1], dtype=np.float64)
    uids = [f"u{i}" for i in range(len(yt))]
    thr = threshold_from_negative_order_statistic(yt, pr, max_fp_clips=1)
    assert abs(thr - 0.7) < 1e-5, thr
    m = clip_metrics_at_fp_budget(uids, yt, pr, max_fp_clips=1)
    assert m["fp"] <= 1


def _check_split() -> None:
    from dataset_cache import assign_splits_stratified_group

    records = []
    for i in range(30):
        records.append(
            {
                "uid": f"neg{i}",
                "group_id": f"g_neg_{i}",
                "is_robbery": False,
                "label": 0,
            }
        )
    for i in range(20):
        records.append(
            {
                "uid": f"rob{i}",
                "group_id": f"g_rob_{i}",
                "is_robbery": True,
                "label": 6,
            }
        )
    ratios = {"train": 0.7, "val_dev": 0.1, "threshold_calib": 0.1, "frozen_test": 0.1}
    mp = assign_splits_stratified_group(records, ratios, seed=42, min_robbery_per_eval_split=5)
    for split in ("val_dev", "threshold_calib", "frozen_test"):
        n_rob = sum(1 for r in records if mp[r["group_id"]] == split and r["is_robbery"])
        assert n_rob >= 5, f"{split} solo {n_rob} robos"


def _check_window_mil() -> None:
    from train_v2.window_labels import window_theft_label

    # clip 96 frames @ 12Hz; ventana corta al inicio no cae en cola de 3s
    y_early = window_theft_label(
        is_robbery_clip=True,
        start_frame=0,
        window_frames=48,
        num_frames=96,
        target_hz=12.0,
        policy="tail_positive",
        tail_seconds=3.0,
    )
    y_late = window_theft_label(
        is_robbery_clip=True,
        start_frame=72,
        window_frames=48,
        num_frames=96,
        target_hz=12.0,
        policy="tail_positive",
        tail_seconds=3.0,
    )
    assert y_early == 0 and y_late == 1


def _check_mil_loss() -> None:
    import torch

    from train_v2.mil_loss import theft_train_loss

    logits = torch.tensor([0.5, 1.0, -1.0, 0.2])
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    uids = ["a", "a", "b", "b"]
    crob = torch.tensor([1.0, 1.0, 0.0, 0.0])
    loss, parts = theft_train_loss(logits, y, uids, crob, bag_loss_weight=0.5)
    assert float(loss.item()) > 0
    assert "window_bce" in parts


def _check_tcn_dual_head() -> None:
    import torch

    from train_v2.models.masked_tcn import MaskedTCNClassifier

    m = MaskedTCNClassifier(40, hidden_dim=16, num_layers=2, aux_concealment=True)
    x = torch.randn(2, 8, 40)
    mask = torch.ones(2, 8)
    out = m(x, mask)
    assert isinstance(out, dict) and "theft" in out and "concealment" in out


def _check_imports() -> None:
    from p0_jobs import resolve_p0_models  # noqa: F401
    from train_v2.calibrate_job import run_p1_calibrate  # noqa: F401
    from train_v2.train_job import run_p1_train  # noqa: F401


def main() -> int:
    checks = [
        ("metrics + umbral T", _check_metrics),
        ("split estratificado (robos en eval)", _check_split),
        ("MIL tail_positive", _check_window_mil),
        ("MIL bag loss", _check_mil_loss),
        ("TCN dual head", _check_tcn_dual_head),
        ("imports P0/P1", _check_imports),
    ]
    for name, fn in checks:
        fn()
        print(f"  OK  {name}")
    print("\nvalidate_campaign2: todo OK (sin GPU, sin mmap)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
