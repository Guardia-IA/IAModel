#!/usr/bin/env python3
"""Estado de jobs P1 (snapshot o watch en el mismo terminal)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CAMPAIGN2_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN2_DIR.parent
for p in (TRAINING_DIR, CAMPAIGN2_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from campaign2_paths import read_current_run_id  # noqa: E402
from live_status import render_live_status, watch_forever  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument(
        "--watch",
        action="store_true",
        help="Refresco en bucle en ESTE terminal (Ctrl+C para salir)",
    )
    ap.add_argument("--interval", type=float, default=4.0, help="Segundos entre refrescos (--watch)")
    args = ap.parse_args()
    run_id = args.run_id or read_current_run_id()
    if not run_id:
        print("ERROR: sin RUN_ID", file=sys.stderr)
        return 2
    if args.watch:
        watch_forever(run_id, interval_sec=args.interval)
        return 0
    print(
        render_live_status(
            run_id,
            title="Campaign2 P1 — estado",
            hint="Durante train el panel se actualiza solo. Opcional: p1-status --watch",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
