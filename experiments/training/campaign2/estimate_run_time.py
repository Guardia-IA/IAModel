#!/usr/bin/env python3
"""Estimación GPU vs CPU para campaña P1 (y referencia legacy)."""
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

from campaign2_paths import (  # noqa: E402
    load_campaign2_config,
    parallel_recommendation_path,
    read_current_run_id,
    run_layout,
)
from manifest_paths import default_manifest_path  # noqa: E402
from time_estimate import build_time_estimate_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Estima tiempos P1 GPU/CPU (heurística ×7)")
    ap.add_argument("--run-id", type=str, default=None, help="Usa manifest del run si existe")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--max-parallel", type=int, default=None)
    args = ap.parse_args()

    cfg = load_campaign2_config(Path(args.config) if args.config else None)
    run_id = args.run_id or read_current_run_id()
    max_par = int(
        args.max_parallel
        if args.max_parallel is not None
        else (cfg.get("parallelism") or {}).get("max_parallel_jobs", 8)
    )
    if args.max_parallel is None and run_id:
        rec = parallel_recommendation_path(run_id)
        if rec.is_file():
            max_par = int(json.loads(rec.read_text()).get("max_parallel_jobs", max_par))

    entries = []
    if run_id:
        mp = default_manifest_path(run_id)
        if mp and mp.is_file():
            entries = json.loads(mp.read_text()).get("entries") or []

    rep = build_time_estimate_report(cfg, manifest_entries=entries, max_parallel_jobs=max_par, run_id=run_id)
    print(f"Estimación P1 train_v2 (factor calibración ×{rep['calibration_factor_applied']:.1f})")
    print(f"  Ventanas/época train/val: {rep['window_train_rows_per_epoch']} / {rep['window_val_rows_per_epoch']}")
    print(f"  Entrenamientos (semillas×jobs plan): {rep['p1_planned_training_runs']}")
    print(f"  Paralelismo asumido: {max_par} jobs")
    print()
    print(f"  GPU secuencial (1 job tras otro): {rep['p1_serial_gpu_human']}")
    print(f"  GPU paralelo×{max_par} (cola):      {rep['p1_parallel_gpu_human']}")
    print(f"  CPU secuencial (referencia):        {rep['p1_serial_cpu_human']}")
    print(f"  CPU paralelo×{max_par} (teórico):   {rep['p1_parallel_cpu_human']}")
    print()
    leg = rep.get("legacy_campaign_20_models_reference")
    if leg:
        from training_time_estimate import fmt_duration

        print(
            f"  Legacy 20 modelos (1 celda, calibrado): sec "
            f"{fmt_duration(leg['serial_seconds_calibrated'])} → "
            f"par×{max_par} {fmt_duration(leg['parallel_seconds'])}"
        )
    print(f"\nNota: {rep.get('note', '')}")
    if run_id:
        out = run_layout(run_id)["reports_dir"] / "time_estimate_gpu_cpu.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
