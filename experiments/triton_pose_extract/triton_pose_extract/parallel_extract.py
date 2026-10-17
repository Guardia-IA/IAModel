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
from triton_pose_extract.discovery import VideoJob
from triton_pose_extract.factory import build_clip_extractor
from triton_pose_extract.settings import ExtractSettings

logger = logging.getLogger(__name__)
_THREAD_EXTRACTOR = threading.local()


def _init_worker(settings: ExtractSettings) -> None:
    _THREAD_EXTRACTOR.extractor = build_clip_extractor(settings)


def _get_extractor() -> Any:
    ext = getattr(_THREAD_EXTRACTOR, "extractor", None)
    if ext is None:
        raise RuntimeError("Worker sin ClipExtractor (fallo init pool)")
    return ext


@dataclass
class ExtractResult:
    video: str
    ok: bool
    out_dir: str | None
    num_users: int
    error: str | None
    seconds: float


def extract_one_job(settings: ExtractSettings, job: VideoJob, out_base: Path) -> ExtractResult:
    t0 = time.perf_counter()
    try:
        ext = _get_extractor()
        cat_folder = str(job.cat) if job.cat is not None else "0"
        result = ext.extract(
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
) -> dict[str, Any]:
    if not jobs:
        return {"ok": False, "error": "sin vídeos", "results": []}

    n_workers = parallel if parallel is not None else recommend_parallel_clips(settings)
    n_workers = max(1, min(n_workers, len(jobs)))
    cap = capacity_report(settings)
    cap["workers_used"] = n_workers

    out_base.mkdir(parents=True, exist_ok=True)
    results: list[ExtractResult] = []
    t0 = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(settings,),
        thread_name_prefix="triton_extract",
    ) as pool:
        futures = {pool.submit(extract_one_job, settings, job, out_base): job for job in jobs}
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
