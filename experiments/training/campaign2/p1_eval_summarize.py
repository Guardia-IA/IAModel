#!/usr/bin/env python3
"""Resumen eval P1 (ventanas, clip-max)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    rows = []
    for d in sorted(layout["jobs_dir"].iterdir()):
        p = d / "p1_eval_result.json"
        if not p.is_file():
            continue
        with open(p, "r", encoding="utf-8") as f:
            r = json.load(f)
        cm = r.get("clip_max_metrics") or {}
        rows.append(
            {
                "eval_job": d.name,
                "source": r.get("source_job_id"),
                "threshold": r.get("threshold"),
                "recall_pct": cm.get("recall_pct"),
                "fp": cm.get("fp"),
                "f1_pct": cm.get("f1_pct"),
                "n_clips": r.get("n_clips"),
            }
        )

    report = {"run_id": run_id, "evaluations": rows}
    out = layout["reports_dir"] / "p1_eval_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"{BOLD}{'═' * 72}{RESET}")
    print(f"{BOLD}P1 eval (clip-max @ umbral){RESET} — {run_id}")
    if not rows:
        print("Sin p1_eval_result.json — ejecuta: ./run_campaign2.sh p1-eval")
        return 0
    for r in rows:
        print(
            f"  {CYAN}{r['source'] or r['eval_job']}{RESET} "
            f"Rec={r.get('recall_pct', 0):.1f}% FP={r.get('fp')} "
            f"F1={r.get('f1_pct', 0):.1f}% (n={r.get('n_clips')})"
        )
    print(f"\nJSON: {out}")
    print(f"{BOLD}{'═' * 72}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
