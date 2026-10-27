#!/usr/bin/env python3
"""Diagnóstico IoU sidecar vs detecciones (1 frame). Requiere pose_core session_close_match (bf03a0f+)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import numpy as np

from triton_pose_extract.extract_stack import build_extract_stack
from triton_pose_extract.settings import ExtractSettings
from triton_pose_extract.video_decode import read_frames_bgr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--subject-json", type=Path, required=True)
    ap.add_argument("--frame", type=int, default=240)
    ap.add_argument("--config", type=str, default="")
    args = ap.parse_args()

    from pose_core.postprocess import decode_end2end, split_triton_batch_output
    from pose_core.preprocess import letterbox, prepare_infer_image, to_tensor
    from pose_core.session_close_match import (
        max_iou_vs_detections,
        pick_session_detection_pre_tracker,
        remap_bbox_norm_c1_live_divisor_to_prep,
        resolve_session_anchor_prep_norm,
    )
    from pose_core.session_subject import load_subject_sidecar, xyxy_pixels_to_norm
    from pose_core.yolo_result import detections_to_mock

    clip = args.clip.expanduser().resolve()
    subj = args.subject_json.expanduser().resolve()
    fi = int(args.frame)

    settings = ExtractSettings.load(
        Path(args.config) if args.config else PKG / "triton_pose_extract_config.json",
        require_pose_repo=True,
    )
    stack = build_extract_stack(settings)
    frames, timing = read_frames_bgr(clip, backend="opencv", decode_c1_pixel_parity=True)
    if fi < 0 or fi >= len(frames):
        print(f"frame {fi} fuera de rango 0..{len(frames)-1}", file=sys.stderr)
        return 2

    sidecar = load_subject_sidecar(subj, num_frames=len(frames), fps=timing.fps or 12.0)
    raw = sidecar.bboxes[fi]
    if raw is None:
        print(f"frame {fi}: sin ancla en sidecar")
        return 0

    frame = frames[fi]
    fh, fw = frame.shape[:2]
    prep = prepare_infer_image(frame, stack.preprocess)
    boxed, lb_meta = letterbox(prep, stack.preprocess)
    tensor = to_tensor(boxed, stack.preprocess)
    raw_out = stack.client.infer(np.stack([tensor], axis=0))
    raw_one = split_triton_batch_output(raw_out, 1)[0]
    mock = detections_to_mock(
        decode_end2end(raw_one, lb_meta, stack.decode_conf),
        orig_shape=(lb_meta.orig_h, lb_meta.orig_w),
    )
    nat = (int(fw), int(fh))
    prep_wh = (lb_meta.orig_w, lb_meta.orig_h)
    imgsz = int(getattr(stack.preprocess, "imgsz", 640))
    min_iou = 0.3

    anchor_prep, remap_tag, frame_iou = resolve_session_anchor_prep_norm(
        raw,
        sidecar=sidecar,
        native_wh=nat,
        prep_wh=prep_wh,
        lb_meta=lb_meta,
        mock_pre=mock,
        imgsz=imgsz,
        min_iou=min_iou,
    )
    c1_anchor = remap_bbox_norm_c1_live_divisor_to_prep(
        raw,
        divisor_w=sidecar.coord_width or nat[0],
        divisor_h=sidecar.coord_height or nat[1],
        prep_w=prep_wh[0],
        prep_h=prep_wh[1],
    )
    pick = pick_session_detection_pre_tracker(
        mock,
        anchor_prep,
        prep_w=prep_wh[0],
        prep_h=prep_wh[1],
        min_iou=min_iou,
        use_sidecar_anchor=True,
    )
    dets = xyxy_pixels_to_norm(mock.boxes.xyxy, prep_wh[0], prep_wh[1])

    out = {
        "frame_index": fi,
        "native_wh": nat,
        "prep_wh": list(prep_wh),
        "sidecar_native_xyxy_norm": raw.tolist(),
        "anchor_c1_live_prep_norm": c1_anchor.tolist(),
        "anchor_resolved_prep_norm": anchor_prep.tolist(),
        "remap_tag": remap_tag,
        "max_iou": frame_iou,
        "max_iou_c1_only": max_iou_vs_detections(
            mock, c1_anchor, prep_w=prep_wh[0], prep_h=prep_wh[1]
        ),
        "pick_index": pick,
        "detections_norm_xyxy": [d.tolist() for d in dets[:5]],
        "coord_width": sidecar.coord_width,
        "coord_height": sidecar.coord_height,
        "bbox_norm_divisor": sidecar.bbox_norm_divisor,
        "bbox_coord_frame": sidecar.bbox_coord_frame,
    }
    print(json.dumps(out, indent=2))
    return 0 if pick is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
