#!/usr/bin/env python3
"""Tabla comparativa tras un batch nocturno (meta JSON + test_metrics)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

WINDOW_DIR = Path(__file__).resolve().parent
if str(WINDOW_DIR) not in sys.path:
    sys.path.insert(0, str(WINDOW_DIR))

from window_paths import load_window_config, models_dir  # noqa: E402


def _dig(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-id", type=str, default=None)
    ap.add_argument("--out-csv", type=str, default=None)
    args = ap.parse_args()

    cfg = load_window_config()
    cell_id = args.cell_id or cfg.get("default_cell_id", "win_bin_full_baseline")
    mdir = models_dir(cell_id)
    rows = []
    for meta_path in sorted(mdir.glob("window_train_meta_*.json")):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        exp_id = int(meta.get("exp_id", 0))
        tm = meta.get("test_metrics") or {}
        bin_m = tm.get("binary") or {}
        rows.append(
            {
                "exp_id": exp_id,
                "arch": meta.get("arch", ""),
                "best_val_acc": meta.get("best_val_acc"),
                "test_acc": tm.get("accuracy"),
                "macro_f1": tm.get("macro_f1"),
                "weighted_f1": tm.get("weighted_f1"),
                "test_f1_pos": _dig(tm, "per_class", "1", "f1"),
                "test_recall_pos": _dig(tm, "per_class", "1", "recall"),
                "binary_f1": bin_m.get("f1"),
                "binary_recall": bin_m.get("recall"),
                "train_seconds": meta.get("train_seconds"),
                "checkpoint": str(mdir / f"modelo_{exp_id:02d}.pt"),
            }
        )

    rows.sort(key=lambda r: r["exp_id"])
    out_csv = Path(args.out_csv) if args.out_csv else mdir / "batch_summary.csv"
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows -> {out_csv}")
        best = max(rows, key=lambda r: float(r.get("macro_f1") or 0))
        print(
            f"Best macro_f1 (clip-window test): exp {best['exp_id']:02d} "
            f"arch={best['arch']} macro_f1={best['macro_f1']}"
        )
    else:
        print(f"No window_train_meta_*.json in {mdir}")


if __name__ == "__main__":
    main()