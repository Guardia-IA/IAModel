#!/usr/bin/env python3
"""Estimación GPU vs CPU para campaña P1 (y referencia legacy)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

CAMPAIGN2_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN2_DIR.parent
for p in (TRAINING_DIR, CAMPAIGN2_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from campaign2_paths import (  # noqa: E402
    load_campaign2_config,
    parallel_recommendation_path,
    read_current_run_id,
    resolve_data_roots,
    run_layout,
)
from dataset_cache import collect_clip_records, manifest_entries_for_time_estimate  # noqa: E402
from manifest_paths import default_manifest_path  # noqa: E402
from matrix_jobs import matrix_settings  # noqa: E402
from time_estimate import build_time_estimate_report, parallel_wall_seconds  # noqa: E402
from training_time_estimate import fmt_duration  # noqa: E402


def _estimate_preflight_wall_seconds(num_clips: int, *, matrix: bool) -> Dict[str, Any]:
    """Heurística empírica (Oct 2026): inventario ~0,03 s/clip; caché 12 Hz ~0,12–0,18 s/clip."""
    n = max(0, int(num_clips))
    inv_scans = 2 if matrix else 1
    caches = 2 if matrix else 1
    inv_s = n * 0.03 * inv_scans
    cache_s = n * 0.15 * caches
    total = inv_s + cache_s
    return {
        "num_clips": n,
        "matrix": matrix,
        "inventory_scans": inv_scans,
        "cache_builds": caches,
        "inventory_seconds": inv_s,
        "cache_seconds": cache_s,
        "total_seconds": total,
        "inventory_human": fmt_duration(inv_s),
        "cache_human": fmt_duration(cache_s),
        "total_human": fmt_duration(total),
        "note": "Solo preflight (inventario + mmap 12 Hz). Matriz prod = 2 scans full+filtered + 2 cachés.",
    }


def _scale_matrix_p1(rep: Dict[str, Any], *, pose_sources: int, max_parallel: int) -> Dict[str, Any]:
    base_runs = int(rep["p1_planned_training_runs"])
    total_runs = base_runs * max(1, pose_sources)
    gpu_serial = float(rep["p1_serial_gpu_seconds"]) * pose_sources
    cpu_serial = float(rep["p1_serial_cpu_seconds"]) * pose_sources
    gpu_par = parallel_wall_seconds(gpu_serial, total_runs, max_parallel)
    cpu_par = parallel_wall_seconds(cpu_serial, total_runs, max_parallel)
    return {
        "pose_sources": pose_sources,
        "p1_training_runs_total": total_runs,
        "p1_serial_gpu_seconds": gpu_serial,
        "p1_parallel_gpu_seconds": gpu_par,
        "p1_serial_cpu_seconds": cpu_serial,
        "p1_parallel_cpu_seconds": cpu_par,
        "p1_serial_gpu_human": fmt_duration(gpu_serial),
        "p1_parallel_gpu_human": fmt_duration(gpu_par),
        "p1_serial_cpu_human": fmt_duration(cpu_serial),
        "p1_parallel_cpu_human": fmt_duration(cpu_par),
    }


def _print_report(
    rep: Dict[str, Any],
    *,
    max_par: int,
    source: str,
    num_clips: int,
    matrix: bool,
    preflight: Optional[Dict[str, Any]],
    matrix_p1: Optional[Dict[str, Any]],
) -> None:
    print(f"Fuente datos: {source}")
    if num_clips:
        print(f"  Clips en inventario/manifest: {num_clips}")
    print(f"Estimación P1 train_v2 (factor calibración ×{rep['calibration_factor_applied']:.1f})")
    print(f"  Ventanas/época train/val: {rep['window_train_rows_per_epoch']} / {rep['window_val_rows_per_epoch']}")
    print(f"  Plan config: {rep['p1_planned_jobs']} jobs × semillas → {rep['p1_planned_training_runs']} entrenamientos")
    print(f"  Paralelismo asumido: {max_par} jobs GPU")
    print()
    if matrix and matrix_p1:
        print(f"  Modo matriz ({matrix_p1['pose_sources']} pose_source: full+filtered)")
        print(f"  Entrenamientos P1 totales: {matrix_p1['p1_training_runs_total']}")
        print(f"  GPU secuencial (todos los trains): {matrix_p1['p1_serial_gpu_human']}")
        print(f"  GPU paralelo×{max_par} (cola):      {matrix_p1['p1_parallel_gpu_human']}")
        print(f"  CPU secuencial (referencia):        {matrix_p1['p1_serial_cpu_human']}")
        print(f"  CPU paralelo×{max_par} (teórico):   {matrix_p1['p1_parallel_cpu_human']}")
    else:
        print(f"  GPU secuencial (1 job tras otro): {rep['p1_serial_gpu_human']}")
        print(f"  GPU paralelo×{max_par} (cola):      {rep['p1_parallel_gpu_human']}")
        print(f"  CPU secuencial (referencia):        {rep['p1_serial_cpu_human']}")
        print(f"  CPU paralelo×{max_par} (teórico):   {rep['p1_parallel_cpu_human']}")
    print()
    leg = rep.get("legacy_campaign_20_models_reference")
    if leg:
        print(
            f"  Legacy 20 modelos (1 celda, calibrado): sec "
            f"{fmt_duration(leg['serial_seconds_calibrated'])} → "
            f"par×{max_par} {fmt_duration(leg['parallel_seconds'])}"
        )
    if preflight and num_clips > 0:
        print()
        print("Preflight (inventario + caché 12 Hz, heurística):")
        print(f"  Inventario ({preflight['inventory_scans']} scan): {preflight['inventory_human']}")
        print(f"  Caché ({preflight['cache_builds']} build):       {preflight['cache_human']}")
        print(f"  Subtotal preflight:                 {preflight['total_human']}")
        if matrix and matrix_p1:
            wall = float(preflight["total_seconds"]) + float(matrix_p1["p1_parallel_gpu_seconds"])
            print(f"  Pared orientativa (preflight + P1 matriz GPU×{max_par}): {fmt_duration(wall)}")
            print("  (+ eval P0/P1, calibrate: suele ser << que el train en dataset grande)")
    print(f"\nNota: {rep.get('note', '')}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Estima tiempos P1 GPU/CPU. Para TODO el dataset usa --data-root (no el RUN debug)."
    )
    ap.add_argument("--run-id", type=str, default=None, help="Opcional: manifest de un run (producción)")
    ap.add_argument("--data-root", action="append", default=[], help="Escaneo inventario completo (repetible)")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--max-parallel", type=int, default=None)
    ap.add_argument(
        "--matrix",
        action="store_true",
        help="Incluye 2× pose_source y tiempos preflight matriz (como matrix-pipeline)",
    )
    args = ap.parse_args()

    cfg = load_campaign2_config(Path(args.config) if args.config else None)
    run_id = args.run_id or (None if args.data_root else read_current_run_id())
    max_par = int(
        args.max_parallel
        if args.max_parallel is not None
        else (cfg.get("parallelism") or {}).get("max_parallel_jobs", 8)
    )
    if args.max_parallel is None and run_id:
        rec = parallel_recommendation_path(run_id)
        if rec.is_file():
            max_par = int(json.loads(rec.read_text()).get("max_parallel_jobs", max_par))

    matrix = bool(args.matrix)
    pose_sources = len(matrix_settings(cfg)["pose_sources"]) if matrix else 1

    entries: List[Dict[str, Any]] = []
    num_clips = 0
    source = "sin datos"

    data_roots = resolve_data_roots(cfg, args.data_root or None)
    if data_roots:
        source = f"inventario completo ({len(data_roots)} raíz/ces)"
        print(f"Escaneando clips (solo metadatos + shape .npy, sin caché)...", flush=True)
        split_cfg = cfg.get("split") or {}
        temporal = cfg.get("temporal") or {}
        records = collect_clip_records(
            data_roots,
            pose_source=str(cfg.get("pose_source", "full")),
            single_user_only=bool(cfg.get("single_user_only", False)),
            robbery_class=int(cfg.get("robbery_class", 6)),
            split_strategy=str(split_cfg.get("strategy", "stratified_group_hybrid")),
            inventory_limit=0,
            log_fn=lambda m: print(m, flush=True),
        )
        num_clips = len(records)
        if not records:
            print("ERROR: 0 clips en --data-root", file=sys.stderr)
            return 2
        print(f"Estimando ventanas/época desde {num_clips} clips...", flush=True)
        entries = manifest_entries_for_time_estimate(
            records,
            target_hz=float(temporal.get("target_hz", 12.0)),
            split_ratios=dict(split_cfg.get("ratios") or {}),
            split_seed=int(split_cfg.get("seed", 42)),
            min_robbery_per_eval_split=int(split_cfg.get("min_robbery_per_eval_split", 5)),
            log_fn=lambda m: print(m, flush=True),
        )
        run_id = None
    elif run_id:
        mp = default_manifest_path(run_id)
        if mp and mp.is_file():
            man = json.loads(mp.read_text(encoding="utf-8"))
            entries = man.get("entries") or []
            num_clips = int(man.get("num_clips") or len(entries))
            source = f"manifest del run {run_id}"
            if num_clips <= 80:
                print(
                    f"AVISO: manifest con solo {num_clips} clips (¿run --debug?). "
                    f"Para estimación real: estimate-time --data-root /ruta/data_clips/ --matrix",
                    flush=True,
                )
        else:
            print(f"WARN: run {run_id} sin manifest; estimación vacía.", flush=True)

    rep = build_time_estimate_report(cfg, manifest_entries=entries, max_parallel_jobs=max_par, run_id=run_id)
    preflight = _estimate_preflight_wall_seconds(num_clips, matrix=matrix) if num_clips else None
    matrix_p1 = _scale_matrix_p1(rep, pose_sources=pose_sources, max_parallel=max_par) if matrix else None

    _print_report(
        rep,
        max_par=max_par,
        source=source,
        num_clips=num_clips,
        matrix=matrix,
        preflight=preflight,
        matrix_p1=matrix_p1,
    )

    out_dir = CAMPAIGN2_DIR / "artifacts" / "reports"
    if run_id:
        out_dir = run_layout(run_id)["reports_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        **rep,
        "estimate_source": source,
        "num_clips": num_clips,
        "matrix_mode": matrix,
        "preflight_heuristic": preflight,
        "matrix_p1": matrix_p1,
    }
    out = out_dir / ("time_estimate_full.json" if data_roots else "time_estimate_gpu_cpu.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
