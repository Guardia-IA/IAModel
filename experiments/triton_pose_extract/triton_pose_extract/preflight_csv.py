"""Preflight CSV → conteos + estimación Triton paralelo."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def scan_csv_inventory(
    csv_files: List[Path],
    pipe: PipelineConfig,
    *,
    probe_videos: bool = False,
) -> Dict[str, Any]:
    category_counters: Dict[str, int] = {}
    by_category: Dict[str, int] = {}
    total = 0
    missing = 0
    skipped_prefilter = 0
    total_seconds = 0.0
    total_frames = 0
    full_without_probe = 0

    for csv_path in csv_files:
        from triton_pose_extract.ffmpeg_cut import video_duration_cv2

        def _dur_fn(p: Path) -> float:
            return video_duration_cv2(p)

        for r in iter_csv_clip_rows(
            csv_path,
            path_root=None,
            category_limits=pipe.category_limits,
            category_counters=category_counters,
            video_duration_fn=_dur_fn,
            end_equals_full_epsilon=pipe.end_equals_full_epsilon,
        ):
            if not r.video_path.is_file():
                missing += 1
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
                continue

            if pipe.filters.min_clip_duration_sec.enabled and dur < pipe.filters.min_clip_duration_sec.value:
                skipped_prefilter += 1
                continue
            if pipe.filters.max_clip_duration_sec.enabled and dur > pipe.filters.max_clip_duration_sec.value:
                skipped_prefilter += 1
                continue

            fps = pipe.estimate_fps
            total += 1
            total_seconds += dur
            cap_dur = min(dur, pipe.filters.max_clip_duration_sec.value if pipe.filters.max_clip_duration_sec.enabled else dur)
            total_frames += int(cap_dur * fps + 0.5)
            by_category[r.category] = by_category.get(r.category, 0) + 1

    return {
        "num_csv": len(csv_files),
        "total_clips": total,
        "by_category": by_category,
        "missing_videos": missing,
        "skipped_prefilter": skipped_prefilter,
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
) -> Dict[str, Any]:
    csv_files = discover_csv_files_many(input_roots)
    inv = scan_csv_inventory(csv_files, pipe, probe_videos=probe_videos)
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


def print_preflight_report(report: Dict[str, Any]) -> None:
    inv = report["inventory"]
    te = report["time_estimate"]
    print("=" * 72)
    print("  TRITON POSE EXTRACT — PREFLIGHT CSV")
    print("=" * 72)
    print("  Carpetas entrada:")
    for p in report.get("input_roots") or []:
        print(f"    - {p}")
    print(f"  CSV encontrados:  {inv['num_csv']}")
    print(f"  Clips a procesar: {inv['total_clips']}")
    print(f"  Vídeos faltantes: {inv['missing_videos']}")
    print(f"  Omitidos (filtros duración): {inv['skipped_prefilter']}")
    print(f"  Salida data_result: {report['output_data_result']}")
    print("\n  Clips por categoría:")
    for cat in sorted(inv["by_category"], key=lambda x: int(x)):
        print(f"    cat {cat:>2}: {inv['by_category'][cat]:>5}")
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


def save_preflight_json(report: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
