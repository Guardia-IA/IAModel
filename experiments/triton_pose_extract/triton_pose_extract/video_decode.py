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
    # ffprobe -count_frames (referencia C1 / gold N2); puede diferir de nb_frames en container
    nb_read_frames: int | None = None
    decode_backend_used: str | None = None


def ffprobe_nb_read_frames(path: Path) -> int | None:
    """Mismo criterio que compare_c1_golden y C1 clip_buffer (T denso)."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=600).strip()
        if out.isdigit():
            n = int(out)
            return n if n > 0 else None
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    return None


def ffprobe_nb_read_packets(path: Path) -> int | None:
    """Conteo rápido (solo demux, sin decodificar). Mismo criterio que C1 ``probe_clip_frame_count``.

    Para MP4 bien formados coincide con ``nb_read_frames``; el llamador debe verificar que el
    decode real produce ese número y, si no, confirmar con ``ffprobe_nb_read_frames``.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=120).strip()
        if out.isdigit():
            n = int(out)
            return n if n > 0 else None
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    return None


def iter_frames_opencv(path: Path):
    """Generador de frames BGR con ``cv2.VideoCapture`` (mismas lecturas y orden que ``_read_opencv``)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


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
    reconcile_ffprobe_frames: bool = True,
    decode_c1_pixel_parity: bool = True,
) -> tuple[list[np.ndarray], VideoTiming]:
    """
    Decodifica todos los frames posibles. Si reconcile_ffprobe_frames (default),
    elige el backend cuya T coincide con ffprobe nb_read_frames (gold C1 N2).
    Con decode_c1_pixel_parity, entre backends con T correcta prioriza OpenCV (C1 reinfer MP4).
    """
    path = Path(path)
    timing = probe_video(path)
    target = ffprobe_nb_read_frames(path) if reconcile_ffprobe_frames else None
    timing = VideoTiming(
        timing.width,
        timing.height,
        timing.fps,
        timing.frame_count,
        timing.codec,
        nb_read_frames=target,
    )

    attempts: list[tuple[str, list[np.ndarray], VideoTiming]] = []

    def _record(name: str, frames: list[np.ndarray], t: VideoTiming) -> None:
        attempts.append(
            (
                name,
                frames,
                VideoTiming(
                    t.width,
                    t.height,
                    t.fps,
                    len(frames),
                    t.codec,
                    nb_read_frames=target,
                    decode_backend_used=name,
                ),
            )
        )

    def _needs_more() -> bool:
        if not reconcile_ffprobe_frames or not target:
            return False
        return not any(len(f) == target for _, f, _ in attempts)

    if backend == "opencv":
        try:
            frames, t = _read_opencv(path, timing)
            _record("opencv", frames, t)
        except Exception as exc:
            logger.warning("opencv falló (%s)", exc)
    elif backend == "ffmpeg_nvdec":
        try:
            frames, t = _read_ffmpeg_nvdec(path, timing, ffmpeg_hwaccel)
            _record("ffmpeg_nvdec", frames, t)
        except Exception as exc:
            logger.warning("ffmpeg_nvdec falló (%s)", exc)
            if not fallback_opencv:
                raise
    else:
        raise ValueError(f"decode backend desconocido: {backend}")

    if _needs_more():
        for name, reader in (
            ("ffmpeg_software", lambda: _read_ffmpeg_software(path, timing)),
            ("ffmpeg_nvdec", lambda: _read_ffmpeg_nvdec(path, timing, ffmpeg_hwaccel)),
            ("opencv", lambda: _read_opencv(path, timing)),
        ):
            if any(n == name for n, _, _ in attempts):
                continue
            try:
                frames, t = reader()
                _record(name, frames, t)
            except Exception as exc:
                logger.warning("%s reconcile falló (%s)", name, exc)
            if not _needs_more():
                break

    if decode_c1_pixel_parity and target:
        # Paridad C1 = OpenCV. Si ya hay un intento OpenCV con T == ffprobe no hace falta
        # decodificar nada más (antes se lanzaba un decode NVDEC extra por clip que luego se
        # descartaba en _pick_decode_attempt). Solo si OpenCV no se intentó, probarlo.
        opencv_exact = any(n == "opencv" and len(f) == target for n, f, _ in attempts)
        if not opencv_exact and not any(n == "opencv" for n, _, _ in attempts):
            try:
                frames, t = _read_opencv(path, timing)
                _record("opencv", frames, t)
            except Exception as exc:
                logger.warning("opencv paridad C1 falló (%s)", exc)

    if not attempts:
        raise RuntimeError(f"decode falló sin frames: {path}")

    chosen_name, chosen_frames, chosen_timing = _pick_decode_attempt(
        attempts, target, prefer_opencv=decode_c1_pixel_parity
    )
    if target and len(chosen_frames) != target:
        logger.warning(
            "%s: decode=%d != ffprobe nb_read_frames=%d (backend=%s)",
            path.name,
            len(chosen_frames),
            target,
            chosen_name,
        )
    elif target:
        logger.info(
            "%s: decode=%d == ffprobe nb_read_frames (%s)",
            path.name,
            len(chosen_frames),
            chosen_name,
        )
    return chosen_frames, chosen_timing


def _pick_decode_attempt(
    attempts: list[tuple[str, list[np.ndarray], VideoTiming]],
    target: int | None,
    *,
    prefer_opencv: bool = True,
) -> tuple[str, list[np.ndarray], VideoTiming]:
    valid = [(n, f, t) for n, f, t in attempts if f]
    if not valid:
        raise RuntimeError("0 frames tras todos los backends")
    if target is not None and target > 0:
        exact = [x for x in valid if len(x[1]) == target]
        if exact:
            if prefer_opencv:
                order = {"opencv": 0, "ffmpeg_software": 1, "ffmpeg_nvdec": 2}
            else:
                order = {"ffmpeg_nvdec": 0, "ffmpeg_software": 1, "opencv": 2}
            exact.sort(key=lambda x: order.get(x[0], 9))
            return exact[0]
    valid.sort(key=lambda x: len(x[1]), reverse=True)
    return valid[0]


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


def _read_ffmpeg_software(path: Path, timing: VideoTiming) -> tuple[list[np.ndarray], VideoTiming]:
    w, h = timing.width, timing.height
    cmd: list[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(path),
        "-an",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
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
        raise RuntimeError(f"ffmpeg software: {err[:350]}")
    if not frames:
        raise RuntimeError("ffmpeg software: 0 frames")
    logger.info("ffmpeg software %s → %d frames", path.name, len(frames))
    return frames, VideoTiming(w, h, timing.fps, len(frames), timing.codec)
