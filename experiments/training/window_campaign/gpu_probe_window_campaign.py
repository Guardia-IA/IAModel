#!/usr/bin/env python3
"""Muestra jobs paralelos recomendados según GPU/VRAM (sin entrenar)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WINDOW_DIR = Path(__file__).resolve().parent
for p in (WINDOW_DIR.parent, WINDOW_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from window_parallel_policy import format_parallel_report, resolve_parallelism
from window_paths import WINDOW_ROOT, load_window_config


def main() -> int:
    ap = argparse.ArgumentParser(description="GPU probe window_campaign")
    ap.add_argument("--parallel", type=int, default=None, help="Forzar N (ignora auto salvo cap)")
    args = ap.parse_args()
    cfg = load_window_config()
    parallel, gpu_cycle, dl, report = resolve_parallelism(cfg, override_parallel=args.parallel)
    report["parallel"] = parallel
    report["gpu_cycle"] = gpu_cycle
    report["dataloader_workers"] = dl
    print(format_parallel_report(report))
    out = WINDOW_ROOT / "artifacts" / "plans" / "gpu_probe_parallel.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())