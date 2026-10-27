#!/usr/bin/env python3
"""Diagnóstico IoU sidecar vs detecciones prep (1 frame o barrido de anclas)."""
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


def _debug_one_frame(
    *,
    stack,
    frames: list,
    sidecar,
    fi: int,
    min_iou: float,
) -> dict:
    from pose_core.postprocess import decode_end2end, split_triton_batch_output
    from pose_core.preprocess import letterbox, prepare_infer_image, to_tensor
    from pose_core.session_close_match import (
        anchor_bbox_prep_norm_candidates,
        max_iou_vs_detections,
        pick_session_detection_pre_tracker,
        remap_bbox_norm_c1_live_divisor_to_prep,
        resolve_session_anchor_prep_norm,
    )
    from pose_core.session_subject import iou_norm, xyxy_pixels_to_norm
    from pose_core.yolo_result import detections_to_mock

    raw = sidecar.bboxes[fi]
    if raw is None:
        return {"frame_index": fi, "skip": "sin ancla"}

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
    div_w = sidecar.coord_width or nat[0]
    div_h = sidecar.coord_height or nat[1]
    c1_anchor = remap_bbox_norm_c1_live_divisor_to_prep(
        raw,
        divisor_w=div_w,
        divisor_h=div_h,
        prep_w=prep_wh[0],
        prep_h=prep_wh[1],
    )
    cands = anchor_bbox_prep_norm_candidates(
        raw,
        sidecar=sidecar,
        native_wh=nat,
        prep_wh=prep_wh,
        lb_meta=lb_meta,
        imgsz=imgsz,
    )
    cand_ious = {
        tag: max_iou_vs_detections(mock, arr, prep_w=prep_wh[0], prep_h=prep_wh[1])
        for tag, arr in cands
    }
    pick = pick_session_detection_pre_tracker(
        mock,
        anchor_prep,
        prep_w=prep_wh[0],
        prep_h=prep_wh[1],
        min_iou=min_iou,
        use_sidecar_anchor=True,
    )
    dets = xyxy_pixels_to_norm(mock.boxes.xyxy, prep_wh[0], prep_wh[1])
    per_det = [
        {"i": j, "bbox_norm": dets[j].tolist(), "iou_vs_anchor": float(iou_norm(dets[j], anchor_prep))}
        for j in range(min(int(dets.shape[0]), 8))
    ]

    return {
        "frame_index": fi,
        "native_wh": nat,
        "prep_wh": list(prep_wh),
        "sidecar_stored_norm_xyxy": raw.tolist(),
        "divisor_wh": [int(div_w), int(div_h)],
        "anchor_c1_live_prep_norm": c1_anchor.tolist(),
        "anchor_resolved_prep_norm": anchor_prep.tolist(),
        "remap_tag_chosen": remap_tag,
        "candidate_max_iou": cand_ious,
        "max_iou_resolved": frame_iou,
        "num_detections": int(dets.shape[0]),
        "per_detection_iou": per_det,
        "pick_index": pick,
        "coord_width": sidecar.coord_width,
        "coord_height": sidecar.coord_height,
        "bbox_norm_divisor": sidecar.bbox_norm_divisor,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--subject-json", type=Path, required=True)
    ap.add_argument("--frame", type=int, default=None, help="Índice de frame (default: primera ancla)")
    ap.add_argument("--scan-anchors", action="store_true", help="Barrido de todos los frames con ancla")
    ap.add_argument("--min-iou", type=float, default=0.3)
    ap.add_argument("--config", type=str, default="")
    args = ap.parse_args()

    from pose_core.session_subject import load_subject_sidecar

    clip = args.clip.expanduser().resolve()
    subj = args.subject_json.expanduser().resolve()

    settings = ExtractSettings.load(
        Path(args.config) if args.config else PKG / "triton_pose_extract_config.json",
        require_pose_repo=True,
    )
    stack = build_extract_stack(settings)
    frames, timing = read_frames_bgr(clip, backend="opencv", decode_c1_pixel_parity=True)
    sidecar = load_subject_sidecar(subj, num_frames=len(frames), fps=timing.fps or 12.0)

    anchor_indices = [i for i, b in enumerate(sidecar.bboxes) if b is not None]
    print(
        json.dumps(
            {
                "clip": str(clip),
                "frames": len(frames),
                "anchor_frames": len(anchor_indices),
                "first_anchor_index": anchor_indices[0] if anchor_indices else None,
                "coord": [sidecar.coord_width, sidecar.coord_height],
                "bbox_norm_divisor": sidecar.bbox_norm_divisor,
            },
            indent=2,
        ),
        flush=True,
    )

    if args.scan_anchors:
        rows = []
        any_pick = False
        for fi in anchor_indices:
            row = _debug_one_frame(
                stack=stack, frames=frames, sidecar=sidecar, fi=fi, min_iou=args.min_iou
            )
            if row.get("pick_index") is not None:
                any_pick = True
            rows.append(
                {
                    "i": fi,
                    "dets": row.get("num_detections"),
                    "max_iou": row.get("max_iou_resolved"),
                    "pick": row.get("pick_index"),
                    "remap": row.get("remap_tag_chosen"),
                }
            )
        print(json.dumps({"scan": rows, "any_pick": any_pick}, indent=2))
        return 0 if any_pick else 1

    fi = args.frame
    if fi is None:
        if not anchor_indices:
            print("sin anclas en sidecar", file=sys.stderr)
            return 2
        fi = anchor_indices[len(anchor_indices) // 2]

    out = _debug_one_frame(stack=stack, frames=frames, sidecar=sidecar, fi=int(fi), min_iou=args.min_iou)
    print(json.dumps(out, indent=2))
    return 0 if out.get("pick_index") is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
