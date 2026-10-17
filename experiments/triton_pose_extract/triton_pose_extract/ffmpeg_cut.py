"""Recorte FFmpeg (misma lógica que pose_extractor_clean, con errores visibles)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Tuple

from triton_pose_extract.pipeline_config import FfmpegConfig


def _hms_to_seconds(t: str) -> float:
    h, m, s = map(int, str(t).split(":"))
    return h * 3600 + m * 60 + s


def _segment_duration_sec(start: str, end: str) -> float:
    return max(0.01, _hms_to_seconds(end) - _hms_to_seconds(start))


def _run_ffmpeg(command: list[str]) -> Tuple[bool, str]:
    result = subprocess.run(command, capture_output=True, text=True)
    err = (result.stderr or result.stdout or "").strip()
    if result.returncode != 0:
        tail = err[-500:] if err else f"exit {result.returncode}"
        return False, tail
    return True, err


def cut_clip(
    video_in: str | Path,
    start: str,
    end: str,
    video_out: str | Path,
    cfg: FfmpegConfig,
) -> Tuple[bool, str]:
    """
    Recorta [start, end). Por defecto (scale_h=0) usa stream copy — rápido;
    pose_core reescala frames para Triton igualmente.
    """
    video_in = str(video_in)
    video_out = str(video_out)
    out_path = Path(video_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scale_h = int(cfg.clip_scale_height or 0)
    vaapi = cfg.vaapi_device
    duration = _segment_duration_sec(start, end)

    def _copy_cmd() -> list[str]:
        return [
            "ffmpeg",
            "-y",
            "-threads",
            "0",
            "-ss",
            str(start),
            "-i",
            video_in,
            "-t",
            f"{duration:.3f}",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            "-loglevel",
            "error",
            str(out_path),
        ]

    def _scale_cmd(use_vaapi: bool) -> list[str]:
        if use_vaapi and vaapi and scale_h:
            return [
                "ffmpeg",
                "-y",
                "-threads",
                "0",
                "-hwaccel",
                "vaapi",
                "-hwaccel_device",
                str(vaapi),
                "-hwaccel_output_format",
                "vaapi",
                "-ss",
                str(start),
                "-i",
                video_in,
                "-t",
                f"{duration:.3f}",
                "-vf",
                f"scale_vaapi=-2:{scale_h}",
                "-c:v",
                "h264_vaapi",
                "-vaapi_device",
                str(vaapi),
                "-loglevel",
                "error",
                str(out_path),
            ]
        return [
            "ffmpeg",
            "-y",
            "-threads",
            "0",
            "-ss",
            str(start),
            "-i",
            video_in,
            "-t",
            f"{duration:.3f}",
            "-vf",
            f"scale=-2:{scale_h}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-loglevel",
            "error",
            str(out_path),
        ]

    attempts: list[tuple[str, list[str]]] = []
    if scale_h <= 0:
        attempts.append(("copy", _copy_cmd()))
    else:
        if vaapi:
            attempts.append(("vaapi", _scale_cmd(True)))
        attempts.append(("libx264", _scale_cmd(False)))
    attempts.append(("copy_fallback", _copy_cmd()))

    last_err = ""
    for label, command in attempts:
        if label == "copy_fallback" and scale_h <= 0:
            continue
        ok, err = _run_ffmpeg(command)
        if ok and out_path.is_file() and out_path.stat().st_size > 0:
            return True, ""
        last_err = f"[{label}] {err}" if err else f"[{label}] sin salida"
        try:
            if out_path.is_file():
                out_path.unlink()
        except OSError:
            pass

    return False, last_err or "ffmpeg sin mensaje"


def video_duration_cv2(video_path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        if fps <= 0 or frames <= 0:
            return 0.0
        return float(frames) / float(fps)
    finally:
        cap.release()
