#!/usr/bin/env python3
"""
Diagnóstico extract gold: decode + Triton + tracker por frame.
Útil cuando el log dice «0 tracks» pero el vídeo sí tiene persona (C1 export OK).

  python3 diagnose_clip_triton.py --video /path/clip_buffer.mp4
  python3 diagnose_clip_triton.py --video ... --tracker none
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))


def _load_config(config: Path | None):
    from extract_clips import _load_raw_config  # noqa: PLC0415
    from triton_pose_extract.pipeline_config import load_pipeline_config  # noqa: PLC0415
    from triton_pose_extract.settings import ExtractSettings  # noqa: PLC0415

    class _Args:
        config = str(config) if config else ""

    raw = _load_raw_config(_Args())
    settings = ExtractSettings.load(
        config or PKG / "triton_pose_extract_config.json",
        require_pose_repo=True,
    )
    pipe = load_pipeline_config(raw, output_base=None)
    return settings, pipe


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnóstico Triton/tracker en un MP4")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--stride", type=int, default=1, help="Procesar 1 de cada N frames")
    ap.add_argument("--max-frames", type=int, default=0, help="Límite frames (0=todos)")
    ap.add_argument(
        "--tracker",
        choices=("bytetrack", "none"),
        default="",
        help="Override tracker (none = 1 detección/frame sin ByteTrack)",
    )
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    settings, pipe = _load_config(args.config)
    from triton_pose_extract.extract_stack import build_extract_stack  # noqa: PLC0415
    from triton_pose_extract.video_decode import read_frames_bgr  # noqa: PLC0415
    from triton_pose_extract.c1_like_extract import _infer_batched  # noqa: PLC0415
    from pose_core.postprocess import decode_end2end  # noqa: PLC0415
    from pose_core.preprocess import letterbox, prepare_infer_image, to_tensor  # noqa: PLC0415
    from pose_core.yolo_result import detections_to_mock, ensure_track_ids  # noqa: PLC0415

    stack = build_extract_stack(settings)
    if args.tracker == "none":
        stack.extract_cfg.tracker_kind = "none"  # type: ignore[misc]

    video = args.video.expanduser().resolve()
    frames, timing = read_frames_bgr(
        video,
        backend=pipe.extract.decode_backend,
        fallback_opencv=pipe.extract.fallback_opencv_decode,
        ffmpeg_hwaccel=pipe.extract.ffmpeg_hwaccel,
    )
    indices = list(range(0, len(frames), max(1, args.stride)))
    if args.max_frames > 0:
        indices = indices[: args.max_frames]

    stats = {
        "video": str(video),
        "decode_backend": pipe.extract.decode_backend,
        "frames_decoded": len(frames),
        "fps": timing.fps,
        "triton_url": settings.triton_url,
        "model_name": settings.model_name,
        "parse_conf": settings.parse_conf,
        "decode_conf": stack.decode_conf,
        "tracker": str(stack.extract_cfg.tracker_kind),
        "input_fp16": settings.input_fp16,
        "frames_sampled": len(indices),
        "frames_with_dets_pre_tracker": 0,
        "frames_with_ids_post_tracker": 0,
        "max_boxes_one_frame": 0,
        "sample_fail_frames": [],
    }

    for fi in indices:
        frame = frames[fi]
        prep = prepare_infer_image(frame, stack.preprocess)
        boxed, lb_meta = letterbox(prep, stack.preprocess)
        tensor = to_tensor(boxed, stack.preprocess)
        raw_out = _infer_batched(stack, [tensor])[0]
        dets = decode_end2end(raw_out, lb_meta, stack.decode_conf)
        n_dets = len(dets) if dets is not None else 0
        if hasattr(dets, "__len__"):
            n_dets = len(dets)
        mock = detections_to_mock(dets, orig_shape=(lb_meta.orig_h, lb_meta.orig_w))
        n_boxes = int(mock.boxes.xyxy.shape[0]) if mock.boxes.xyxy is not None else 0
        if n_boxes > 0:
            stats["frames_with_dets_pre_tracker"] += 1
            stats["max_boxes_one_frame"] = max(stats["max_boxes_one_frame"], n_boxes)

        if stack.extract_cfg.tracker_kind == "none":
            from triton_pose_extract.c1_like_extract import _none_tracker_pick  # noqa: PLC0415

            mock = _none_tracker_pick(mock)
        else:
            mock = stack.tracker.update(mock, prep)
        mock = ensure_track_ids(mock)
        ids_raw = getattr(mock.boxes, "id", None)
        has_ids = ids_raw is not None and n_boxes > 0
        if has_ids:
            stats["frames_with_ids_post_tracker"] += 1
        elif n_boxes > 0 and len(stats["sample_fail_frames"]) < 15:
            stats["sample_fail_frames"].append(
                {"frame_index": fi, "n_boxes": n_boxes, "ids": None if ids_raw is None else "set"}
            )

    stats["verdict"] = _verdict(stats)
    text = json.dumps(stats, indent=2, ensure_ascii=False)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if stats["frames_with_ids_post_tracker"] > 0 else 2


def _verdict(stats: dict) -> str:
    if stats["frames_decoded"] == 0:
        return "DECODE_VACIO: revisar codec/NVDEC/OpenCV"
    if stats["frames_with_dets_pre_tracker"] == 0:
        return (
            "TRITON_SIN_DETECCIONES: modelo/preprocess/FP16 o GPU saturada "
            "(para batch extract-csv en paralelo)"
        )
    if stats["frames_with_ids_post_tracker"] == 0:
        return "TRACKER_SIN_IDS: probar --tracker none o revisar yolo26s-pose.pt / ByteTrack"
    return "OK: hay frames con track_id; si extract sigue users=0, revisar filtros BODY_VISIBLE_*"


if __name__ == "__main__":
    sys.exit(main())
