#!/usr/bin/env python3
"""Reimprime el resumen bonito desde preflight_summary.json (sin re-ejecutar preflight)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CAMPAIGN2_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN2_DIR.parent
for p in (TRAINING_DIR, CAMPAIGN2_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from campaign2_paths import artifacts_root, load_campaign2_config, read_current_run_id  # noqa: E402
from preflight_display import render_preflight_summary  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--json", type=str, default=None, help="Ruta a preflight_summary.json")
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()

    cfg = load_campaign2_config(Path(args.config) if args.config else None)
    if args.json:
        path = Path(args.json)
    else:
        run_id = args.run_id or read_current_run_id()
        if not run_id:
            print("ERROR: --run-id o --json", file=sys.stderr)
            return 2
        path = artifacts_root(run_id) / "reports" / "preflight_summary.json"

    if not path.is_file():
        print(f"ERROR: no existe {path}", file=sys.stderr)
        return 2

    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)
    report["_json_path"] = str(path)
    print(render_preflight_summary(report, robbery_class=int(cfg.get("robbery_class", 6))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
