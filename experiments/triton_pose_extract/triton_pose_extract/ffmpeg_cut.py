"""Recorte FFmpeg (misma lógica que pose_extractor_clean)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from triton_pose_extract.pipeline_config import FfmpegConfig


def cut_clip(video_in: str | Path, start: str, end: str, video_out: str | Path, cfg: FfmpegConfig) -> bool:
    video_in = str(video_in)
    video_out = str(video_out)
    scale_h = cfg.clip_scale_height
    vaapi = cfg.vaapi_device

    if vaapi and scale_h:
        scale_expr = f"scale_vaapi=-2:{scale_h}"
        command = [
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
            "-to",
            str(end),
            "-i",
            video_in,
            "-vf",
            scale_expr,
            "-c:v",
            "h264_vaapi",
            "-vaapi_device",
            str(vaapi),
            "-loglevel",
            "error",
            video_out,
        ]
    elif scale_h:
        command = [
            "ffmpeg",
            "-y",
            "-threads",
            "0",
            "-ss",
            str(start),
            "-to",
            str(end),
            "-i",
            video_in,
            "-vf",
            f"scale=-2:{scale_h}",
            "-c:v",
            "libx264",
            "-loglevel",
            "error",
            video_out,
        ]
    else:
        command = [
            "ffmpeg",
            "-y",
            "-threads",
            "0",
            "-ss",
            str(start),
            "-to",
            str(end),
            "-i",
            video_in,
            "-c",
            "copy",
            "-loglevel",
            "error",
            video_out,
        ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 and vaapi and scale_h:
        command_fb = [
            "ffmpeg",
            "-y",
            "-threads",
            "0",
            "-ss",
            str(start),
            "-to",
            str(end),
            "-i",
            video_in,
            "-vf",
            f"scale=-2:{scale_h}",
            "-c:v",
            "libx264",
            "-loglevel",
            "error",
            video_out,
        ]
        result = subprocess.run(command_fb, capture_output=True, text=True)
    return result.returncode == 0 and os.path.isfile(video_out)


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
