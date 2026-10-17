"""Preflight CSV → conteos + estimación Triton paralelo."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from triton_pose_extract.capacity import capacity_report, recommend_parallel_clips
from triton_pose_extract.csv_io import discover_csv_files_many, iter_csv_clip_rows
from triton_pose_extract.pipeline_config import PipelineConfig
from triton_pose_extract.settings import ExtractSettings


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def _log(msg: str, log_fn: Optional[Callable[[str], None]] = None) -> None:
    fn = log_fn or (lambda s: print(s, flush=True))
    fn(msg)


def scan_csv_inventory(
    csv_files: List[Path],
    pipe: PipelineConfig,
    *,
    probe_videos: bool = False,
    measure_video_duration: bool = False,
    verbose: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
    collect_missing_paths: bool = False,
) -> Dict[str, Any]:
    """
    probe_videos: ffprobe en clips marcados como completos (lento, preciso).
    measure_video_duration: abre cada vídeo con OpenCV para duración (MUY lento).
    Por defecto solo HMS del CSV + estimación 10s en 00:00:00–00:00:00.
    """
    category_counters: Dict[str, int] = {}
    by_category: Dict[str, int] = {}
    total = 0
    missing = 0
    skipped_prefilter = 0
    skipped_too_short = 0
    skipped_too_long = 0
    skipped_invalid_duration = 0
    missing_entries: List[Dict[str, Any]] = []
    total_seconds = 0.0
    total_frames = 0
    full_without_probe = 0
    rows_seen = 0
    missing_cap = 5000 if collect_missing_paths else 0

    duration_fn = None
    if measure_video_duration:
        from triton_pose_extract.ffmpeg_cut import video_duration_cv2

        cache: Dict[str, float] = {}

        def duration_fn(p: Path) -> float:
            key = str(p.resolve())
            if key not in cache:
                cache[key] = video_duration_cv2(p)
            return cache[key]

    n_csv = len(csv_files)
    t0 = time.perf_counter()
    for csv_idx, csv_path in enumerate(csv_files, start=1):
        if verbose:
            _log(f"[{csv_idx}/{n_csv}] CSV {csv_path}", log_fn)
        n_clip_before = total
        rows_in_file = 0

        for r in iter_csv_clip_rows(
            csv_path,
            path_root=None,
            category_limits=pipe.category_limits,
            category_counters=category_counters,
            video_duration_fn=duration_fn,
            end_equals_full_epsilon=pipe.end_equals_full_epsilon,
        ):
            rows_in_file += 1
            rows_seen += 1
            if verbose and rows_seen % 2000 == 0:
                _log(f"  … {rows_seen} filas CSV leídas, {total} clips válidos", log_fn)
            if not r.video_path.is_file():
                missing += 1
                if len(missing_entries) < missing_cap:
                    missing_entries.append(
                        {
                            "csv": str(r.csv_path),
                            "row": r.row_csv,
                            "category": r.category,
                            "video_csv": r.video_rel,
                            "path_resolved": str(r.video_path),
                        }
                    )
                continue

            dur = r.duration_sec
            if r.use_full_clip:
                if probe_videos:
                    try:
                        out = subprocess.run(
                            [
                                "ffprobe",
                                "-v",
                                "error",
                                "-show_entries",
                                "format=duration",
                                "-of",
                                "csv=p=0",
                                str(r.video_path),
                            ],
                            capture_output=True,
                            text=True,
                            timeout=10,
                            check=False,
                        )
                        dur = float(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else 0.0
                    except Exception:
                        dur = 0.0
                else:
                    dur = pipe.estimate_full_clip_sec
                    full_without_probe += 1
            elif dur <= 0:
                skipped_invalid_duration += 1
                continue

            if pipe.filters.min_clip_duration_sec.enabled and dur < pipe.filters.min_clip_duration_sec.value:
                skipped_prefilter += 1
                skipped_too_short += 1
                continue
            if pipe.filters.max_clip_duration_sec.enabled and dur > pipe.filters.max_clip_duration_sec.value:
                skipped_prefilter += 1
                skipped_too_long += 1
                continue

            fps = pipe.estimate_fps
            total += 1
            total_seconds += dur
            cap_dur = min(dur, pipe.filters.max_clip_duration_sec.value if pipe.filters.max_clip_duration_sec.enabled else dur)
            total_frames += int(cap_dur * fps + 0.5)
            by_category[r.category] = by_category.get(r.category, 0) + 1

        if verbose:
            added = total - n_clip_before
            _log(f"  → {rows_in_file} filas, {added} clips válidos en este CSV", log_fn)

    if verbose:
        _log(f"Inventario CSV en {time.perf_counter() - t0:.1f}s", log_fn)

    return {
        "num_csv": len(csv_files),
        "csv_rows_seen": rows_seen,
        "total_clips": total,
        "by_category": by_category,
        "missing_videos": missing,
        "skipped_prefilter": skipped_prefilter,
        "skipped_too_short": skipped_too_short,
        "skipped_too_long": skipped_too_long,
        "skipped_invalid_duration": skipped_invalid_duration,
        "missing_entries": missing_entries,
        "missing_entries_truncated": missing > len(missing_entries),
        "total_seconds_video": total_seconds,
        "total_frames_est": total_frames,
        "full_clip_without_probe": full_without_probe,
        "probe_videos": probe_videos,
    }


def estimate_wall_time(
    inv: Dict[str, Any],
    pipe: PipelineConfig,
    settings: ExtractSettings,
) -> Dict[str, Any]:
    par = recommend_parallel_clips(settings)
    clips = int(inv["total_clips"])
    frames = int(inv["total_frames_est"])
    infer_s = frames * pipe.triton_sec_per_frame
    ffmpeg_s = clips * pipe.ffmpeg_sec_per_clip
    serial = infer_s + ffmpeg_s
    parallel = ffmpeg_s / par + infer_s / par
    return {
        "parallel_clips": par,
        "capacity": capacity_report(settings),
        "infer_seconds": round(infer_s, 1),
        "ffmpeg_seconds": round(ffmpeg_s, 1),
        "wall_serial_seconds": round(serial, 1),
        "wall_parallel_seconds": round(parallel, 1),
        "wall_serial_human": _fmt_duration(serial),
        "wall_parallel_human": _fmt_duration(parallel),
        "triton_sec_per_frame": pipe.triton_sec_per_frame,
        "ffmpeg_sec_per_clip": pipe.ffmpeg_sec_per_clip,
    }


def run_preflight_report(
    input_roots: List[Path],
    pipe: PipelineConfig,
    settings: ExtractSettings,
    *,
    probe_videos: bool = False,
    measure_video_duration: bool = False,
    verbose: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
    collect_missing_paths: bool = False,
) -> Dict[str, Any]:
    _log("Preflight CSV — buscando *.csv …", log_fn)
    for root in input_roots:
        _log(f"  raíz: {root}", log_fn)
    t_disc = time.perf_counter()
    csv_files = discover_csv_files_many(input_roots)
    _log(f"Encontrados {len(csv_files)} CSV ({time.perf_counter() - t_disc:.1f}s)", log_fn)
    if not csv_files:
        _log("AVISO: ningún CSV; revisa --input-root", log_fn)
    if probe_videos:
        _log("Modo: ffprobe en clips completos (más lento)", log_fn)
    elif measure_video_duration:
        _log("Modo: duración por OpenCV por vídeo (muy lento)", log_fn)
    else:
        _log(
            "Modo: rápido (tiempos del CSV; 00:00:00–00:00:00 ≈ "
            f"{pipe.estimate_full_clip_sec}s). Usa --probe-videos para precisión.",
            log_fn,
        )
    inv = scan_csv_inventory(
        csv_files,
        pipe,
        probe_videos=probe_videos,
        measure_video_duration=measure_video_duration,
        verbose=verbose,
        log_fn=log_fn,
        collect_missing_paths=collect_missing_paths,
    )
    time_est = estimate_wall_time(inv, pipe, settings)
    return {
        "input_roots": [str(p) for p in input_roots],
        "csv_files": [str(p) for p in csv_files],
        "inventory": inv,
        "time_estimate": time_est,
        "output_data_result": str(pipe.paths.data_result_dir),
        "filters": {
            "single_user_only": pipe.filters.single_user_only,
            "min_clip_duration_sec": pipe.filters.min_clip_duration_sec.__dict__,
            "max_clip_duration_sec": pipe.filters.max_clip_duration_sec.__dict__,
            "skip_existing_output": pipe.filters.skip_existing_output,
        },
    }


def print_preflight_report(report: Dict[str, Any], *, list_missing: int = 0) -> None:
    inv = report["inventory"]
    te = report["time_estimate"]
    rows = int(inv.get("csv_rows_seen") or 0)
    ok = int(inv["total_clips"])
    miss = int(inv["missing_videos"])
    sk = int(inv["skipped_prefilter"])
    sk_inv = int(inv.get("skipped_invalid_duration") or 0)
    accounted = ok + miss + sk + sk_inv

    print("=" * 72)
    print("  TRITON POSE EXTRACT — PREFLIGHT CSV")
    print("=" * 72)
    print("  Carpetas entrada:")
    for p in report.get("input_roots") or []:
        print(f"    - {p}")
    print(f"  CSV encontrados:     {inv['num_csv']}")
    print(f"  Filas CSV (válidas): {rows}")
    print("")
    print(f"  >>> TOTAL CLIPS A PROCESAR: {ok} <<<")
    print(f"      (suma por categoría:    {sum(inv['by_category'].values())})")
    print("")
    print("  Desglose de filas no procesables:")
    print(f"    Vídeos faltantes:        {miss}  (ruta columna 0 no existe en disco)")
    print(f"    Duración inválida:       {sk_inv}")
    print(f"    Filtro duración (total): {sk}")
    if sk:
        print(f"      · demasiado cortos:    {inv.get('skipped_too_short', 0)}")
        print(f"      · demasiado largos:    {inv.get('skipped_too_long', 0)}")
    if rows and accounted != rows:
        print(f"    (Nota: {accounted} contabilizadas vs {rows} filas; resto = límites cat. en CSV)")
    print(f"\n  Salida data_result: {report['output_data_result']}")
    print("\n  Clips a procesar por categoría:")
    for cat in sorted(inv["by_category"], key=lambda x: int(x)):
        print(f"    cat {cat:>2}: {inv['by_category'][cat]:>5}")
    print(f"    {'─' * 20}")
    print(f"    {'TOTAL':>6}: {sum(inv['by_category'].values()):>5}")
    mins = inv["total_seconds_video"] / 60.0
    print(f"\n  Duración vídeo (estimada): {mins:.1f} min")
    print(f"  Frames estimados: {inv['total_frames_est']}")
    if inv.get("full_clip_without_probe"):
        print(
            f"  ({inv['full_clip_without_probe']} clips completos contados con "
            f"{report.get('estimate_full_clip_sec', '?')}s sin ffprobe; usa --probe-videos)"
        )
    print("\n  Paralelismo Triton:")
    print(f"    workers: {te['parallel_clips']}  ({te['capacity']['formula']})")
    print(f"    inferencia ≈ {_fmt_duration(te['infer_seconds'])}")
    print(f"    FFmpeg ≈ {_fmt_duration(te['ffmpeg_seconds'])}")
    print(f"    TOTAL pared ≈ {te['wall_parallel_human']}  (serial ≈ {te['wall_serial_human']})")
    print("=" * 72)

    if list_missing > 0 and inv.get("missing_entries"):
        print(f"\n  Ejemplos vídeos faltantes (máx {list_missing}):")
        for ent in inv["missing_entries"][:list_missing]:
            print(f"    fila {ent['row']} cat={ent['category']}")
            print(f"      CSV: {ent['video_csv'][:100]}{'…' if len(ent['video_csv']) > 100 else ''}")
            print(f"      →   {ent['path_resolved']}")
        if miss > list_missing:
            print(f"    … y {miss - list_missing} más")


def save_preflight_json(report: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
