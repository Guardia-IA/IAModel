"""Ejecuta cola CSV → FFmpeg → Triton (paralelo)."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional  # noqa: F401

from triton_pose_extract.capacity import recommend_parallel_clips
from triton_pose_extract.csv_jobs import ExtractClipJob, build_jobs_from_csvs, resolve_csv_list
from triton_pose_extract.factory import build_clip_extractor
from triton_pose_extract.ffmpeg_cut import cut_clip, video_duration_cv2
from triton_pose_extract.pipeline_config import PipelineConfig
from triton_pose_extract.settings import ExtractSettings
from triton_pose_extract.validate_output import validate_clip_dir

logger = logging.getLogger(__name__)
_THREAD_EXTRACTOR = threading.local()


def _init_worker(settings: ExtractSettings) -> None:
    _THREAD_EXTRACTOR.extractor = build_clip_extractor(settings)


def _get_extractor():
    ext = getattr(_THREAD_EXTRACTOR, "extractor", None)
    if ext is None:
        raise RuntimeError("Worker sin ClipExtractor")
    return ext


@dataclass
class JobResult:
    clip_name: str
    ok: bool
    out_dir: Optional[str]
    error: Optional[str]
    seconds: float
    csv_row: int


def _process_one_job(job: ExtractClipJob, pipe: PipelineConfig, settings: ExtractSettings) -> JobResult:
    t0 = time.perf_counter()
    r = job.row
    try:
        if job.clip_is_temp:
            if not job.work_video.is_file():
                ok = cut_clip(
                    r.video_path,
                    r.ffmpeg_start,
                    r.ffmpeg_end,
                    job.work_video,
                    pipe.ffmpeg,
                )
                if not ok:
                    return JobResult(
                        job.clip_name,
                        False,
                        None,
                        "ffmpeg_falló",
                        time.perf_counter() - t0,
                        r.row_csv,
                    )

        ext = _get_extractor()
        job.data_dir.parent.mkdir(parents=True, exist_ok=True)
        result = ext.extract(
            job.work_video,
            out_dir=job.data_dir.parent,
            cat=r.category,
            clip_folder=job.clip_name,
        )
        meta_path = result.out_dir / "meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["video_source"] = str(r.video_path.resolve())
            meta["row_csv"] = int(r.row_csv)
            meta["t_start"] = r.t_start
            meta["t_end"] = r.t_end
            meta["full_clip"] = r.use_full_clip
            meta["csv"] = str(r.csv_path)
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        errs = validate_clip_dir(result.out_dir)
        if errs:
            return JobResult(
                job.clip_name,
                False,
                str(result.out_dir),
                ";".join(errs),
                time.perf_counter() - t0,
                r.row_csv,
            )

        if pipe.ffmpeg.delete_temp_videos and job.clip_is_temp and job.work_video.is_file():
            try:
                os.remove(job.work_video)
            except OSError:
                pass

        return JobResult(
            job.clip_name,
            True,
            str(result.out_dir),
            None,
            time.perf_counter() - t0,
            r.row_csv,
        )
    except Exception as exc:
        logger.exception("job %s", job.clip_name)
        return JobResult(
            job.clip_name,
            False,
            None,
            str(exc),
            time.perf_counter() - t0,
            r.row_csv,
        )


def run_csv_batch(
    input_roots: List[Path],
    pipe: PipelineConfig,
    settings: ExtractSettings,
    *,
    parallel: int | None = None,
    limit: int = 0,
) -> Dict[str, Any]:
    csv_files = resolve_csv_list(input_roots)
    jobs, skipped, by_cat = build_jobs_from_csvs(
        csv_files,
        pipe,
        path_root=None,
        limit=limit,
    )
    if not jobs:
        return {
            "ok": False,
            "error": "sin_jobs",
            "skipped": [asdict(s) for s in skipped[:50]],
            "num_csv": len(csv_files),
        }

    n_workers = parallel if parallel is not None else recommend_parallel_clips(settings)
    n_workers = max(1, min(n_workers, len(jobs)))
    results: List[JobResult] = []
    t0 = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(settings,),
        thread_name_prefix="csv_triton",
    ) as pool:
        futs = {pool.submit(_process_one_job, job, pipe, settings): job for job in jobs}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            tag = "OK" if res.ok else "FAIL"
            print(
                f"[{tag}] {res.clip_name} fila={res.csv_row} {res.seconds:.1f}s"
                + (f" {res.error}" if res.error else ""),
                flush=True,
            )

    ok_n = sum(1 for r in results if r.ok)
    report = {
        "ok": ok_n > 0,
        "input_roots": [str(p) for p in input_roots],
        "num_csv": len(csv_files),
        "jobs_planned": len(jobs),
        "jobs_ok": ok_n,
        "jobs_failed": len(jobs) - ok_n,
        "workers": n_workers,
        "by_category_planned": by_cat,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "data_result": str(pipe.paths.data_result_dir),
        "results": [asdict(r) for r in results],
        "skipped_count": len(skipped),
    }
    out_report = pipe.paths.output_base / "logs" / "csv_extract_report.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Informe: {out_report}", flush=True)
    return report
