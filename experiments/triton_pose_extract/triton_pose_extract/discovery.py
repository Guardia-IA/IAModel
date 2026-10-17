"""Descubrimiento de vídeos (data_result / carpetas sueltas)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoJob:
    video_path: Path
    cat: str | None
    clip_folder: str | None


def discover_videos(
    *,
    video: Path | None = None,
    video_dir: Path | None = None,
    cat: str | None = None,
    limit: int = 0,
) -> list[VideoJob]:
    jobs: list[VideoJob] = []
    if video is not None:
        vpath = video.expanduser().resolve()
        cf = vpath.parent.name if vpath.name == "clip.mp4" else vpath.stem
        jobs.append(VideoJob(vpath, cat, cf))
    if video_dir is not None:
        root = video_dir.expanduser().resolve()
        resolve_clip_cat = None
        try:
            from pose_core.golden_layout import resolve_clip_cat as _rcc

            resolve_clip_cat = _rcc
        except ImportError:
            pass

        for mp4 in sorted(root.rglob("clip.mp4")):
            clip_dir = mp4.parent
            if resolve_clip_cat is not None:
                c = resolve_clip_cat(clip_dir, root)
            else:
                c = clip_dir.parent.name
            jobs.append(VideoJob(mp4, str(c) if c is not None else cat, clip_dir.name))
        if not jobs:
            for mp4 in sorted(root.rglob("*.mp4")):
                jobs.append(VideoJob(mp4, cat, mp4.stem))
    if limit > 0:
        jobs = jobs[:limit]
    return jobs
