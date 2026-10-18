"""Estimación de batch a partir de muestra real (más fiable que preflight)."""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def fmt_duration(seconds: float) -> str:
    if seconds < 0 or not (seconds < 1e12):
        return "?"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


@dataclass
class SampleTimingStats:
    n_total: int
    n_ok: int
    n_skip: int
    n_fail: int
    wall_seconds: float
    workers: int
    seconds_all: list[float]
    seconds_ok: list[float]
    median_ok: float
    mean_ok: float
    p90_ok: float
    throughput_jobs_per_hour: float


def stats_from_job_results(
    results: list[Any],
    *,
    wall_seconds: float,
    workers: int,
) -> SampleTimingStats:
    """results: objetos con .seconds, .status (o .ok)."""
    seconds_all: list[float] = []
    seconds_ok: list[float] = []
    n_ok = n_skip = n_fail = 0
    for r in results:
        sec = float(getattr(r, "seconds", 0) or 0)
        seconds_all.append(sec)
        st = getattr(r, "status", None)
        if st is None and getattr(r, "ok", False):
            st = "ok"
        if st == "ok":
            n_ok += 1
            seconds_ok.append(sec)
        elif st == "fail_infra":
            n_fail += 1
        else:
            n_skip += 1

    seconds_ok.sort()
    median_ok = statistics.median(seconds_ok) if seconds_ok else 0.0
    mean_ok = statistics.mean(seconds_ok) if seconds_ok else 0.0
    p90_ok = _percentile(seconds_ok, 90.0) if seconds_ok else 0.0
    n_total = len(results)
    tph = (n_total / wall_seconds * 3600.0) if wall_seconds > 0 else 0.0
    return SampleTimingStats(
        n_total=n_total,
        n_ok=n_ok,
        n_skip=n_skip,
        n_fail=n_fail,
        wall_seconds=wall_seconds,
        workers=workers,
        seconds_all=seconds_all,
        seconds_ok=seconds_ok,
        median_ok=median_ok,
        mean_ok=mean_ok,
        p90_ok=p90_ok,
        throughput_jobs_per_hour=tph,
    )


@dataclass
class BatchProjection:
    pending_jobs: int
    inventory_clips: int
    pending_source: str
    eta_seconds_throughput: float
    eta_seconds_median_ok: float
    yield_ok_pct_sample: float


def project_batch(
    stats: SampleTimingStats,
    *,
    pending_jobs: int,
    inventory_clips: int,
    pending_source: str,
) -> BatchProjection:
    yield_pct = (100.0 * stats.n_ok / stats.n_total) if stats.n_total else 0.0
    eta_tp = 0.0
    if stats.throughput_jobs_per_hour > 0 and pending_jobs > 0:
        eta_tp = pending_jobs / (stats.throughput_jobs_per_hour / 3600.0)
    eta_med = 0.0
    if stats.median_ok > 0 and pending_jobs > 0 and stats.workers > 0:
        eta_med = pending_jobs * stats.median_ok / stats.workers
    return BatchProjection(
        pending_jobs=pending_jobs,
        inventory_clips=inventory_clips,
        pending_source=pending_source,
        eta_seconds_throughput=eta_tp,
        eta_seconds_median_ok=eta_med,
        yield_ok_pct_sample=yield_pct,
    )


def print_bench_report(
    stats: SampleTimingStats,
    proj: BatchProjection,
    *,
    extract_engine: str = "",
    extra_lines: list[str] | None = None,
) -> None:
    print("")
    print("=" * 72)
    print("  BENCH EXTRACT — muestra real")
    print("=" * 72)
    if extract_engine:
        print(f"  Motor: {extract_engine}")
    print(f"  Muestra: {stats.n_total} jobs · workers={stats.workers} · pared={fmt_duration(stats.wall_seconds)}")
    print(f"  OK={stats.n_ok}  SKIP={stats.n_skip}  FAIL={stats.n_fail}  "
          f"(yield npy ≈ {proj.yield_ok_pct_sample:.1f}%)")
    print("")
    if stats.seconds_ok:
        print("  Tiempo por clip (solo OK):")
        print(f"    mediana: {stats.median_ok:.1f}s   media: {stats.mean_ok:.1f}s   p90: {stats.p90_ok:.1f}s")
    else:
        print("  AVISO: ningún OK en la muestra; revisa Triton/tracker/filtros.")
    print("")
    print(f"  Throughput del pool: {stats.throughput_jobs_per_hour:.1f} jobs/h "
          f"({stats.n_total / stats.wall_seconds:.3f} jobs/s)" if stats.wall_seconds > 0 else "")
    print("")
    print("  Proyección batch:")
    print(f"    Clips inventario CSV (rápido): {proj.inventory_clips}")
    print(f"    Pendientes estimados:          {proj.pending_jobs}  ({proj.pending_source})")
    if proj.pending_jobs > 0 and stats.wall_seconds > 0:
        print(f"    ETA (throughput muestra):      {fmt_duration(proj.eta_seconds_throughput)}")
        if proj.eta_seconds_median_ok > 0:
            print(
                f"    ETA (mediana OK × jobs/workers): {fmt_duration(proj.eta_seconds_median_ok)}  "
                "(modelo alternativo)"
            )
    print("")
    print("  Notas:")
    print("    · La muestra respeta skip_existing (primeros N pendientes en cola).")
    print("    · SKIP (multi-track/single_user) no produce npy pero consume tiempo.")
    print("    · Para pending exacto usa bench-csv --count-pending (construye cola completa, lento).")
    if extra_lines:
        for line in extra_lines:
            print(f"    · {line}")
    print("=" * 72)


def save_bench_json(report: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def analyze_report_file(report_path: Path, *, pending_jobs: int | None = None) -> None:
    """Re-analiza csv_extract_report.json o bench_report.json."""
    data = json.loads(report_path.read_text(encoding="utf-8"))
    results = data.get("results") or []

    class _R:
        pass

    objs = []
    for row in results:
        o = _R()
        o.seconds = row.get("seconds", 0)
        o.status = row.get("status")
        o.ok = row.get("ok")
        o.error = row.get("error")
        objs.append(o)
    wall = float(data.get("wall_seconds") or 0)
    workers = int(data.get("workers") or 1)
    stats = stats_from_job_results(objs, wall_seconds=wall, workers=workers)
    inv = int(data.get("inventory_clips") or data.get("jobs_planned") or 0)
    pend = pending_jobs if pending_jobs is not None else int(data.get("pending_jobs") or inv)
    proj = project_batch(
        stats,
        pending_jobs=pend,
        inventory_clips=inv,
        pending_source=str(data.get("pending_source") or "desde informe"),
    )
    print_bench_report(stats, proj, extract_engine=str(data.get("extract_engine") or ""))
