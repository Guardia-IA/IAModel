"""Extracción paralela: un ClipExtractor (y cliente gRPC) por hilo."""
from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from triton_pose_extract.capacity import capacity_report, recommend_parallel_clips
from triton_pose_extract.c1_like_extract import extract_clip_c1_like
from triton_pose_extract.discovery import VideoJob
from triton_pose_extract.factory import build_pose_stack
from triton_pose_extract.pipeline_config import (
    C1LikeExtractConfig,
    PipelineConfig,
    PipelineFilters,
)
from triton_pose_extract.settings import ExtractSettings

logger = logging.getLogger(__name__)
_THREAD = threading.local()


def _init_worker(settings: ExtractSettings, use_c1_like: bool) -> None:
    stack = build_pose_stack(settings)
    _THREAD.stack = stack
    _THREAD.use_c1_like = use_c1_like
    if not use_c1_like:
        from pose_core.clip_extractor import ClipExtractor

        _THREAD.extractor = ClipExtractor(
            stack.client,
            stack.preprocess,
            stack.tracker,
            stack.bundle,
            extract_cfg=stack.extract_cfg,
        )


def _get_extractor() -> Any:
    ext = getattr(_THREAD, "extractor", None)
    if ext is None:
        raise RuntimeError("Worker sin ClipExtractor (fallo init pool)")
    return ext


def _get_stack() -> Any:
    st = getattr(_THREAD, "stack", None)
    if st is None:
        raise RuntimeError("Worker sin PoseExtractStack (fallo init pool)")
    return st


@dataclass
class ExtractResult:
    video: str
    ok: bool
    out_dir: str | None
    num_users: int
    error: str | None
    seconds: float


def extract_one_job(
    settings: ExtractSettings,
    job: VideoJob,
    out_base: Path,
    *,
    pipe_extract: C1LikeExtractConfig | None = None,
    pipe_filters: PipelineFilters | None = None,
) -> ExtractResult:
    t0 = time.perf_counter()
    try:
        cat_folder = str(job.cat) if job.cat is not None else "0"
        if getattr(_THREAD, "use_c1_like", False):
            if pipe_extract is None:
                raise RuntimeError("c1_like requiere pipe.extract")
            result, _timings = extract_clip_c1_like(
                _get_stack(),
                job.video_path,
                out_dir=out_base / cat_folder,
                cat=cat_folder,
                clip_folder=job.clip_folder,
                pipe_cfg=pipe_extract,
                filters=pipe_filters,
            )
        else:
            stack = _get_stack()
            if pipe_filters and pipe_filters.subject_bbox_json:
                stack.extract_cfg.session_close = True
                stack.extract_cfg.subject_track_json = pipe_filters.subject_bbox_json
            else:
                stack.extract_cfg.session_close = False
                stack.extract_cfg.subject_track_json = None
            result = _get_extractor().extract(
                job.video_path,
                out_dir=out_base / cat_folder,
                cat=cat_folder,
                clip_folder=job.clip_folder,
            )
        return ExtractResult(
            video=str(job.video_path),
            ok=True,
            out_dir=str(result.out_dir),
            num_users=len(result.user_dirs),
            error=None,
            seconds=time.perf_counter() - t0,
        )
    except Exception as exc:
        logger.exception("extract failed %s", job.video_path)
        return ExtractResult(
            video=str(job.video_path),
            ok=False,
            out_dir=None,
            num_users=0,
            error=str(exc),
            seconds=time.perf_counter() - t0,
        )


def run_parallel_extract(
    settings: ExtractSettings,
    jobs: list[VideoJob],
    out_base: Path,
    *,
    parallel: int | None = None,
    pipe: PipelineConfig | None = None,
) -> dict[str, Any]:
    if not jobs:
        return {"ok": False, "error": "sin vídeos", "results": []}

    n_workers = parallel if parallel is not None else recommend_parallel_clips(settings)
    n_workers = max(1, min(n_workers, len(jobs)))
    cap = capacity_report(settings)
    cap["workers_used"] = n_workers

    use_c1_like = bool(pipe and pipe.extract.use_c1_like())
    pipe_extract = pipe.extract if pipe else None

    out_base.mkdir(parents=True, exist_ok=True)
    results: list[ExtractResult] = []
    t0 = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(settings, use_c1_like),
        thread_name_prefix="triton_extract",
    ) as pool:
        futures = {
            pool.submit(
                extract_one_job,
                settings,
                job,
                out_base,
                pipe_extract=pipe_extract,
                pipe_filters=pipe.filters if pipe else None,
            ): job
            for job in jobs
        }
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            tag = "OK" if res.ok else "FAIL"
            print(
                f"[{tag}] {res.video} users={res.num_users} {res.seconds:.1f}s"
                + (f" err={res.error}" if res.error else f" → {res.out_dir}"),
                flush=True,
            )

    ok_n = sum(1 for r in results if r.ok)
    report = {
        "ok": ok_n > 0,
        "extract_engine": "c1_like" if use_c1_like else "clip_extractor",
        "capacity": cap,
        "num_jobs": len(jobs),
        "num_ok": ok_n,
        "num_failed": len(jobs) - ok_n,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "out_base": str(out_base),
        "results": [asdict(r) for r in results],
    }
    report_path = out_base / "_extract_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Informe: {report_path}", flush=True)
    return report
