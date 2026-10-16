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


def _check_imports() -> None:
    from p0_jobs import resolve_p0_models  # noqa: F401
    from train_v2.calibrate_job import run_p1_calibrate  # noqa: F401
    from train_v2.train_job import run_p1_train  # noqa: F401


def main() -> int:
    checks = [
        ("metrics + umbral T", _check_metrics),
        ("split estratificado (robos en eval)", _check_split),
        ("imports P0/P1", _check_imports),
    ]
    for name, fn in checks:
        fn()
        print(f"  OK  {name}")
    print("\nvalidate_campaign2: todo OK (sin GPU, sin mmap)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
