#!/usr/bin/env python3
"""Tabla final tras entrenamiento P1."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

CAMPAIGN2_DIR = Path(__file__).resolve().parent
for p in (CAMPAIGN2_DIR.parent, CAMPAIGN2_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from campaign2_paths import read_current_run_id, run_layout  # noqa: E402

BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
RESET = "\033[0m"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    args = ap.parse_args()
    run_id = args.run_id or read_current_run_id()
    if not run_id:
        print("ERROR: sin RUN_ID", file=sys.stderr)
        return 2

    layout = run_layout(run_id)
    rows: List[Dict[str, Any]] = []
    for d in sorted(layout["jobs_dir"].iterdir()):
        if not d.is_dir():
            continue
        p = d / "p1_train_result.json"
        if not p.is_file():
            continue
        with open(p, "r", encoding="utf-8") as f:
            r = json.load(f)
        le = r.get("last_epoch") or {}
        rows.append(
            {
                "job_id": d.name,
                "arch": r.get("arch"),
                "seed": r.get("seed"),
                "epochs": r.get("epochs_ran"),
                "best_score": r.get("best_score"),
                "val_recall_pct": le.get("val_clip_recall_pct"),
                "val_fp": le.get("val_clip_fp"),
                "val_fp_rate_pct": le.get("val_clip_fp_rate_pct"),
                "checkpoint": r.get("best_checkpoint"),
            }
        )

    report = {"run_id": run_id, "jobs": rows}
    out = layout["reports_dir"] / "p1_train_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"{BOLD}{'═' * 72}{RESET}")
    print(f"{BOLD}P1 entrenamiento — resumen{RESET} ({run_id})")
    print(f"{BOLD}{'═' * 72}{RESET}")
    if not rows:
        print("Sin p1_train_result.json — ¿train terminado?")
        return 0
    best = max(rows, key=lambda x: float(x.get("best_score") or -1e9))
    for r in rows:
        mark = f"{GREEN}★{RESET} " if r["job_id"] == best["job_id"] else "  "
        print(
            f"{mark}{CYAN}{r['job_id']}{RESET} seed={r['seed']} "
            f"rec={r.get('val_recall_pct', 0):.1f}% FP={r.get('val_fp')} "
            f"score={r.get('best_score', 0):.2f}"
        )
    print(f"\nMejor job: {GREEN}{best['job_id']}{RESET}")
    print(f"Checkpoint: {best.get('checkpoint')}")
    print(f"JSON: {out}")
    print(f"{BOLD}{'═' * 72}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
