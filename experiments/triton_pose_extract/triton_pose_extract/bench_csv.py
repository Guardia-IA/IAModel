"""Bench: procesar N clips y estimar ETA del batch completo."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

from triton_pose_extract.batch_csv import run_csv_batch
from triton_pose_extract.bench_estimate import (
    fmt_duration,
    print_bench_report,
    project_batch,
    save_bench_json,
    stats_from_job_results,
)
from triton_pose_extract.csv_jobs import build_jobs_from_csvs, resolve_csv_list
from triton_pose_extract.pipeline_config import PipelineConfig
from triton_pose_extract.preflight_csv import scan_csv_inventory
from triton_pose_extract.settings import ExtractSettings


def run_bench_csv(
    input_roots: List[Path],
    pipe: PipelineConfig,
    settings: ExtractSettings,
    *,
    sample: int = 50,
    parallel: int | None = None,
    count_pending: bool = False,
    with_inventory: bool = False,
    total_clips: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    if sample <= 0:
        raise ValueError("sample debe ser > 0")

    csv_files = resolve_csv_list(input_roots)
    inventory_clips = int(total_clips or 0)
    pending_jobs = inventory_clips
    pending_source = "total_clips manual (--total-clips) o desconocido"

    if with_inventory:
        print("Bench: inventario CSV (22k filas en 8TB puede tardar ~7 min)…", flush=True)
        t0 = time.perf_counter()
        inv = scan_csv_inventory(csv_files, pipe, verbose=False)
        inventory_clips = int(inv["total_clips"])
        pending_jobs = inventory_clips
        pending_source = "inventario CSV (no resta skip_existing ni cola exacta)"
        print(f"  {len(csv_files)} CSV · {inventory_clips} clips ({time.perf_counter() - t0:.1f}s)", flush=True)
    elif inventory_clips > 0:
        pending_source = f"--total-clips {inventory_clips} (sin escaneo CSV)"
        print(f"Bench: proyección sobre {inventory_clips} clips (sin inventario previo)", flush=True)
    else:
        print(
            "Bench: sin inventario (rápido). Proyección batch: pasa --total-clips 22409 o --with-inventory",
            flush=True,
        )
        pending_source = "desconocido; añade --total-clips para ETA"

    if count_pending:
        print(
            "Bench: construyendo cola completa (OpenCV duración + skip_existing) — puede tardar mucho…",
            flush=True,
        )
        t1 = time.perf_counter()
        all_jobs, skipped, _ = build_jobs_from_csvs(csv_files, pipe, limit=0, verbose=verbose)
        pending_jobs = len(all_jobs)
        pending_source = f"cola real build_jobs ({len(skipped)} filas saltadas en prefilter)"
        print(f"  Pendientes en cola: {pending_jobs} ({time.perf_counter() - t1:.1f}s)", flush=True)

    print(f"Bench: extracción muestra n={sample} …", flush=True)
    extract_report = run_csv_batch(
        input_roots,
        pipe,
        settings,
        parallel=parallel,
        limit=sample,
        verbose=verbose,
    )

    wall = float(extract_report.get("wall_seconds") or 0)
    workers = int(extract_report.get("workers") or 1)

    class _Wrap:
        pass

    wrapped = []
    for row in extract_report.get("results") or []:
        w = _Wrap()
        w.seconds = row.get("seconds", 0)
        w.status = row.get("status")
        w.ok = row.get("ok")
        w.error = row.get("error")
        wrapped.append(w)

    stats = stats_from_job_results(wrapped, wall_seconds=wall, workers=workers)

    if not count_pending and pipe.filters.skip_existing_output:
        pending_source += "; usa --count-pending para restar ya exportados"

    proj = project_batch(
        stats,
        pending_jobs=pending_jobs,
        inventory_clips=inventory_clips,
        pending_source=pending_source,
    )

    engine = "c1_like" if pipe.extract.use_c1_like() else "clip_extractor"
    if pipe.extract.use_c1_like():
        engine = f"c1_like ({pipe.extract.decode_backend}, prep={pipe.extract.preprocess_workers})"

    print_bench_report(stats, proj, extract_engine=engine)

    bench_doc = {
        "bench_sample_size": sample,
        "count_pending_used": count_pending,
        "inventory_clips": inventory_clips,
        "pending_jobs": pending_jobs,
        "pending_source": pending_source,
        "extract_engine": engine,
        "sample_stats": {
            "n_total": stats.n_total,
            "n_ok": stats.n_ok,
            "n_skip": stats.n_skip,
            "n_fail": stats.n_fail,
            "wall_seconds": stats.wall_seconds,
            "workers": stats.workers,
            "median_ok_sec": round(stats.median_ok, 3),
            "mean_ok_sec": round(stats.mean_ok, 3),
            "p90_ok_sec": round(stats.p90_ok, 3),
            "throughput_jobs_per_hour": round(stats.throughput_jobs_per_hour, 2),
        },
        "projection": {
            "eta_throughput_sec": round(proj.eta_seconds_throughput, 1),
            "eta_throughput_human": fmt_duration(proj.eta_seconds_throughput),
            "eta_median_model_sec": round(proj.eta_seconds_median_ok, 1),
            "eta_median_model_human": fmt_duration(proj.eta_seconds_median_ok),
            "yield_ok_pct_sample": proj.yield_ok_pct_sample,
        },
        "extract_run": extract_report,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = pipe.paths.output_base / "logs" / f"bench_extract_{ts}.json"
    save_bench_json(bench_doc, out_path)
    print(f"Informe bench: {out_path}", flush=True)
    return bench_doc
