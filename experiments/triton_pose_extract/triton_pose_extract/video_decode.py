"""Decode MP4 → frames BGR (NVDEC vía FFmpeg o OpenCV). Solo IAModel."""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_CUVID = {"h264": "h264_cuvid", "avc": "h264_cuvid", "hevc": "hevc_cuvid", "h265": "hevc_cuvid"}


@dataclass
class VideoTiming:
    width: int
    height: int
    fps: float
    frame_count: int
    codec: str


def probe_video(path: Path) -> VideoTiming:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name,nb_frames,r_frame_rate,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=60)
        st = (json.loads(out.decode()) or {}).get("streams") or [{}]
        st = st[0]
        w, h = int(st["width"]), int(st["height"])
        codec = str(st.get("codec_name") or "h264").lower()
        nb = st.get("nb_frames")
        n = int(nb) if nb not in (None, "N/A", "") else 0
        fps = _parse_fps(st.get("avg_frame_rate") or st.get("r_frame_rate") or "12/1")
        if n <= 0 and fps > 0:
            dur = subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                timeout=30,
            )
            n = max(1, int(round(float(dur.decode().strip() or 0) * fps)))
        return VideoTiming(w, h, fps, n, codec)
    except (FileNotFoundError, subprocess.SubprocessError, (KeyError, ValueError, json.JSONDecodeError)):
        return _probe_opencv(path)


def read_frames_bgr(
    path: Path,
    *,
    backend: str,
    fallback_opencv: bool = True,
    ffmpeg_hwaccel: str = "cuda",
) -> tuple[list[np.ndarray], VideoTiming]:
    path = Path(path)
    timing = probe_video(path)
    if backend == "opencv":
        return _read_opencv(path, timing)
    if backend == "ffmpeg_nvdec":
        try:
            return _read_ffmpeg_nvdec(path, timing, ffmpeg_hwaccel)
        except Exception as exc:
            logger.warning("ffmpeg_nvdec falló (%s)", exc)
            if not fallback_opencv:
                raise
            return _read_opencv(path, timing)
    raise ValueError(f"decode backend desconocido: {backend}")


def _parse_fps(raw: str) -> float:
    raw = str(raw).strip()
    if "/" in raw:
        a, b = raw.split("/", 1)
        d = float(b or 1)
        return float(a) / d if d else 12.0
    try:
        v = float(raw)
        return v if v > 0 else 12.0
    except ValueError:
        return 12.0


def _probe_opencv(path: Path) -> VideoTiming:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir: {path}")
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 12.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return VideoTiming(w, h, fps, n, "unknown")


def _read_opencv(path: Path, timing: VideoTiming) -> tuple[list[np.ndarray], VideoTiming]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"0 frames: {path}")
    timing = VideoTiming(timing.width, timing.height, timing.fps, len(frames), timing.codec)
    return frames, timing


def _read_ffmpeg_nvdec(path: Path, timing: VideoTiming, hwaccel: str) -> tuple[list[np.ndarray], VideoTiming]:
    w, h = timing.width, timing.height
    cuvid = _CUVID.get(timing.codec)
    cmd: list[str] = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if cuvid:
        if hwaccel:
            cmd += ["-hwaccel", hwaccel]
        cmd += ["-c:v", cuvid]
    cmd += ["-i", str(path), "-an", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    fb = w * h * 3
    frames: list[np.ndarray] = []
    while True:
        buf = proc.stdout.read(fb)
        if not buf or len(buf) < fb:
            break
        frames.append(np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).copy())
    err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    if proc.wait() != 0 and not frames:
        raise RuntimeError(f"ffmpeg: {err[:350]}")
    if not frames:
        raise RuntimeError("ffmpeg nvdec: 0 frames")
    logger.info("nvdec %s → %d frames", path.name, len(frames))
    return frames, VideoTiming(w, h, timing.fps, len(frames), timing.codec)
