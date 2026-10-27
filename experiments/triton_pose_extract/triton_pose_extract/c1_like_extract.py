"""
Extracción tipo C1 en IAModel: NVDEC + preprocess paralelo → Triton (pose_core público).
No modifica kanvis-triton-pose; replica el bucle de ClipExtractor usando sus primitivas.
"""
from __future__ import annotations

import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from triton_pose_extract.extract_stack import PoseExtractStack
from triton_pose_extract.pipeline_config import C1LikeExtractConfig
from triton_pose_extract.video_decode import read_frames_bgr

logger = logging.getLogger(__name__)

DEFAULT_FPS = 12.0
NAN_POSE = np.full((8, 2), np.nan, dtype=np.float32)
EMPTY_KP_MASK = np.zeros(8, dtype=bool)


@dataclass
class _UserAccum:
    poses_full: list[np.ndarray]
    poses: list[np.ndarray]
    keypoint_masks: list[np.ndarray]
    body_visible_cnt: int = 0
    total: int = 0


def extract_clip_c1_like(
    stack: PoseExtractStack,
    video_path: str | Path,
    *,
    out_dir: str | Path,
    cat: str | int | None = None,
    clip_folder: str | None = None,
    pipe_cfg: C1LikeExtractConfig,
    filters: Any | None = None,
) -> tuple[Any, dict[str, float]]:
    from pose_core.clip_extractor import ClipPoseResult
    from pose_core.export import (
        PersonRows,
        build_masked_pose_from_xyn,
        frame_valid,
        keypoint_mask,
        write_person_artifacts,
    )
    from pose_core.postprocess import decode_end2end
    from pose_core.preprocess import letterbox, prepare_infer_image, to_tensor
    from pose_core.yolo_result import detections_to_mock, ensure_track_ids

    video_path = Path(video_path)
    out_dir = Path(out_dir)
    clip_name = video_path.stem
    folder = (clip_folder or clip_name).strip() or clip_name
    clip_root = out_dir / folder
    clip_root.mkdir(parents=True, exist_ok=True)

    import time

    t0 = time.perf_counter()
    frames, timing = read_frames_bgr(
        video_path,
        backend=pipe_cfg.decode_backend,
        fallback_opencv=pipe_cfg.fallback_opencv_decode,
        ffmpeg_hwaccel=pipe_cfg.ffmpeg_hwaccel,
        decode_c1_pixel_parity=pipe_cfg.decode_c1_pixel_parity,
    )
    t_decode = time.perf_counter() - t0

    fps = timing.fps if timing.fps > 0 else DEFAULT_FPS
    frame_count = timing.frame_count or len(frames)
    cfg = stack.extract_cfg

    native_frame_wh: list[tuple[int, int]] = []

    def _prep_one(frame: np.ndarray):
        fh, fw = frame.shape[:2]
        native_frame_wh.append((int(fw), int(fh)))
        prep = prepare_infer_image(frame, stack.preprocess)
        boxed, lb_meta = letterbox(prep, stack.preprocess)
        return prep, lb_meta, to_tensor(boxed, stack.preprocess)

    t1 = time.perf_counter()
    workers = pipe_cfg.preprocess_workers
    if workers <= 1 or len(frames) <= 1:
        prepared = [_prep_one(f) for f in frames]
    else:
        prepared = [None] * len(frames)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="c1_prep") as pool:
            futs = {pool.submit(_prep_one, frames[i]): i for i in range(len(frames))}
            for fut in futs:
                prepared[futs[fut]] = fut.result()
    t_prep = time.perf_counter() - t1

    prepared_list = [p[0] for p in prepared]
    meta_list = [p[1] for p in prepared]
    tensors = [p[2] for p in prepared]

    stack.tracker.reset()
    users: dict[int, _UserAccum] = {}

    t2 = time.perf_counter()
    raw_batches = _infer_batched(stack, tensors)
    decoded_t = len(prepared_list)

    subject_json = getattr(filters, "subject_bbox_json", None) if filters else None
    if subject_json:
        result, timings = _export_session_subject_iou(
            stack=stack,
            clip_root=clip_root,
            clip_name=clip_name,
            video_path=video_path,
            prepared_list=prepared_list,
            meta_list=meta_list,
            raw_batches=raw_batches,
            subject_json=subject_json,
            pipe_cfg=pipe_cfg,
            fps=fps,
            frame_count=frame_count,
            timing=timing,
            decoded_t=decoded_t,
            native_frame_wh=native_frame_wh,
            cat=cat,
            cfg=cfg,
            t0=t0,
            t_decode=t_decode,
            t_prep=t_prep,
            t2=t2,
        )
        return result, timings

    use_keep_all = cfg.frame_policy == "keep_all"
    for frame_index, (prep, lb_meta, raw_out) in enumerate(
        zip(prepared_list, meta_list, raw_batches, strict=True)
    ):
        dets = decode_end2end(raw_out, lb_meta, stack.decode_conf)
        mock = detections_to_mock(dets, orig_shape=(lb_meta.orig_h, lb_meta.orig_w))
        if cfg.tracker_kind == "none":
            mock = _none_tracker_pick(mock)
        else:
            mock = stack.tracker.update(mock, prep)
        mock = ensure_track_ids(mock)
        if use_keep_all:
            _process_frame_keep_all(users, mock, cfg, frame_index)
        else:
            _accumulate_frame(users, mock, cfg, frame_policy=cfg.frame_policy)

    if use_keep_all:
        _assert_keep_all_timeline(users, decoded_t, clip_name)

    session_close = bool(getattr(filters, "session_close_export", False))
    export_track_id = getattr(filters, "export_track_id", None)
    if export_track_id is not None:
        export_track_id = int(export_track_id)
    relax_quality = session_close or export_track_id is not None

    if cfg.single_user_only and len(users) > 1 and not relax_quality:
        logger.warning(
            "clip %s: single_user_only y %d tracks → sin export (usa export_primary_user "
            "o --session-close)",
            clip_name,
            len(users),
        )
        users = {}

    if export_track_id is not None and export_track_id not in users:
        logger.warning(
            "clip %s: export_track_id=%d no visto en ByteTrack (tracks=%s)",
            clip_name,
            export_track_id,
            sorted(users.keys()),
        )

    user_dirs: list[Path] = []
    meta_users: list[dict] = []
    filtered = 0
    for tid, acc in sorted(users.items()):
        if export_track_id is not None and int(tid) != export_track_id:
            continue
        if not relax_quality and not _passes_filters(acc, frame_count):
            filtered += 1
            continue
        user_dir = clip_root / f"user_{tid}"
        poses_t = len(acc.poses_full)
        if use_keep_all and poses_t != decoded_t:
            logger.error(
                "clip %s user_%s: poses_full T=%d != decoded_T=%d (no exportar)",
                clip_name,
                tid,
                poses_t,
                decoded_t,
            )
            filtered += 1
            continue
        write_person_artifacts(
            user_dir,
            PersonRows(poses_full=acc.poses_full, poses=acc.poses, keypoint_masks=acc.keypoint_masks),
        )
        user_dirs.append(user_dir)
        valid_frames = sum(1 for p in acc.poses_full if frame_valid(p, min_visible=5))
        passed_quality = _passes_filters(acc, frame_count)
        meta_users.append(
            {
                "track_id": int(tid),
                "valid_pct": round(100.0 * valid_frames / max(decoded_t, 1), 2),
                "valid_frames": valid_frames,
                "total_frames": poses_t,
                "poses_full_T": poses_t,
                "passes_filters": passed_quality,
                "export_relaxed": relax_quality and not passed_quality,
            }
        )

    if users and not user_dirs:
        logger.warning(
            "clip %s: %d track(s) filtros (filtered=%d frames=%d)",
            clip_name,
            len(users),
            filtered,
            len(prepared_list),
        )
    elif not users:
        logger.warning("clip %s: 0 tracks", clip_name)

    if cfg.save_clip_mp4:
        dest_mp4 = clip_root / "clip.mp4"
        if video_path.resolve() != dest_mp4.resolve():
            shutil.copy2(video_path, dest_mp4)

    cat_val = str(cat) if cat is not None else "0"
    ffprobe_t = timing.nb_read_frames
    exported_pose_t = (
        int(meta_users[0]["poses_full_T"]) if meta_users else 0
    )
    meta = {
        "clip_name": clip_name,
        "fps": fps,
        "frame_count": decoded_t,
        "poses_full_T": exported_pose_t or decoded_t,
        "frame_policy": cfg.frame_policy,
        "video_frame_count": frame_count,
        "ffprobe_nb_read_frames": ffprobe_t,
        "decode_backend_used": timing.decode_backend_used or pipe_cfg.decode_backend,
        "frame_count_matches_ffprobe": (
            ffprobe_t is not None and len(prepared_list) == int(ffprobe_t)
        ),
        "cat": cat_val,
        "clip_duration": round(len(prepared_list) / fps, 3) if fps > 0 else 0,
        "yolo_backend": "triton_tensorrt_plan",
        "pose_bundle_id": stack.bundle.get("pose_bundle_id", ""),
        "users": meta_users,
        "clip_video": "clip.mp4",
        "iamodel_extract_engine": "c1_like",
        "iamodel_decode_backend": pipe_cfg.decode_backend,
        "iamodel_preprocess_workers": pipe_cfg.preprocess_workers,
        "iamodel_session_close_export": session_close,
        "iamodel_export_track_id": export_track_id,
    }
    (clip_root / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (clip_root / "pose_bundle.json").write_text(
        json.dumps(stack.bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    t_infer = time.perf_counter() - t2
    timings = {
        "decode_sec": round(t_decode, 3),
        "preprocess_sec": round(t_prep, 3),
        "infer_track_export_sec": round(t_infer, 3),
        "total_sec": round(time.perf_counter() - t0, 3),
    }
    result = ClipPoseResult(clip_name=clip_name, out_dir=clip_root, meta=meta, user_dirs=user_dirs)
    return result, timings


def _export_session_subject_iou(
    *,
    stack: PoseExtractStack,
    clip_root: Path,
    clip_name: str,
    video_path: Path,
    prepared_list: list,
    meta_list: list,
    raw_batches: list[np.ndarray],
    subject_json: str,
    pipe_cfg: C1LikeExtractConfig,
    fps: float,
    frame_count: int,
    timing: Any,
    decoded_t: int,
    native_frame_wh: list[tuple[int, int]],
    cat: str | int | None,
    cfg: Any,
    t0: float,
    t_decode: float,
    t_prep: float,
    t2: float,
) -> tuple[Any, dict[str, float]]:
    """Misma lógica que ClipExtractor.session_close + subject_track (pose_core)."""
    import time

    from pose_core.clip_extractor import ClipPoseResult
    from pose_core.export import build_masked_pose_from_xyn, keypoint_mask, write_session_close_artifacts
    from pose_core.postprocess import decode_end2end
    from pose_core.session_close_match import (
        pick_session_detection_pre_tracker,
        resolve_session_anchor_prep_norm,
    )
    from pose_core.session_subject import (
        load_subject_sidecar,
        picked_bbox_norm,
        subject_sidecar_loaded,
        xyxy_pixels_to_norm,
    )
    from pose_core.yolo_result import detections_to_mock

    min_iou = float(getattr(cfg, "session_close_min_iou", 0.3))
    uid = int(getattr(cfg, "session_close_user_id", 0))
    min_kp = float(getattr(cfg, "min_kp_conf", 0.5))

    sidecar_path = Path(subject_json)
    has_sidecar = subject_sidecar_loaded(subject_json)
    sidecar_info = load_subject_sidecar(subject_json, num_frames=decoded_t, fps=fps)
    subject_bboxes = sidecar_info.bboxes
    imgsz = int(getattr(stack.preprocess, "imgsz", 640) or 640)
    debug_frames: set[int] = set()
    if filters is not None:
        debug_frames = set(getattr(filters, "debug_subject_frames", None) or [])
    if has_sidecar:
        logger.info(
            "session_close sidecar session_id=%s bbox_space=%s anchors=%s/%s path=%s",
            sidecar_info.session_id,
            sidecar_info.bbox_space,
            sidecar_info.anchor_nonnull,
            len(subject_bboxes),
            sidecar_path.name,
        )
    session_poses: list[np.ndarray] = []
    session_kms: list[np.ndarray] = []
    sc_stats = {
        "frames_anchor_nonnull": 0,
        "frames_with_detections": 0,
        "frames_pick_iou_ok": 0,
        "frames_pose_any_kp": 0,
        "max_iou_anchor_sample": 0.0,
        "anchor_remap_c1_live_frames": 0,
        "anchor_remap_geometric_frames": 0,
    }
    max_iou_seen = 0.0
    prev_subject_bbox: np.ndarray | None = None

    for prep, lb_meta, raw_out in zip(
        prepared_list, meta_list, raw_batches, strict=True
    ):
        dets = decode_end2end(raw_out, lb_meta, stack.decode_conf)
        mock_pre = detections_to_mock(dets, orig_shape=(lb_meta.orig_h, lb_meta.orig_w))

        frame_idx = len(session_poses)
        raw_anchor = (
            subject_bboxes[frame_idx] if frame_idx < len(subject_bboxes) else None
        )
        nat_w, nat_h = (
            native_frame_wh[frame_idx]
            if frame_idx < len(native_frame_wh)
            else (lb_meta.orig_w, lb_meta.orig_h)
        )
        prep_w, prep_h = lb_meta.orig_w, lb_meta.orig_h

        anchor_prep = None
        if raw_anchor is not None:
            sc_stats["frames_anchor_nonnull"] += 1
            anchor_prep, remap_tag, frame_iou = resolve_session_anchor_prep_norm(
                raw_anchor,
                sidecar=sidecar_info,
                native_wh=(nat_w, nat_h),
                prep_wh=(prep_w, prep_h),
                lb_meta=lb_meta,
                mock_pre=mock_pre,
                imgsz=imgsz,
                min_iou=min_iou,
            )
            max_iou_seen = max(max_iou_seen, frame_iou)
            if remap_tag == "c1_live":
                sc_stats["anchor_remap_c1_live_frames"] += 1
            else:
                sc_stats["anchor_remap_geometric_frames"] += 1
            if frame_idx in debug_frames and anchor_prep is not None:
                dets_n = xyxy_pixels_to_norm(mock_pre.boxes.xyxy, prep_w, prep_h)[:3]
                logger.warning(
                    "subject_debug frame=%s native_wh=%s prep_wh=%s "
                    "anchor_native=%s anchor_prep=%s max_iou=%.4f remap=%s dets_norm=%s",
                    frame_idx,
                    (nat_w, nat_h),
                    (prep_w, prep_h),
                    raw_anchor.tolist(),
                    anchor_prep.tolist(),
                    frame_iou,
                    remap_tag,
                    [d.tolist() for d in dets_n],
                )

        n_det = int(mock_pre.boxes.xyxy.shape[0])
        if n_det > 0:
            sc_stats["frames_with_detections"] += 1

        pick = pick_session_detection_pre_tracker(
            mock_pre,
            anchor_prep,
            prep_w=prep_w,
            prep_h=prep_h,
            min_iou=min_iou,
            prev_bbox_norm=prev_subject_bbox,
            use_sidecar_anchor=has_sidecar,
        )
        if has_sidecar and raw_anchor is None:
            prev_subject_bbox = None
        elif pick is not None:
            prev_subject_bbox = picked_bbox_norm(mock_pre, pick, prep_w, prep_h)

        if pick is None:
            session_poses.append(NAN_POSE.copy())
            session_kms.append(EMPTY_KP_MASK.copy())
        else:
            sc_stats["frames_pick_iou_ok"] += 1
            xyn = mock_pre.keypoints.xyn
            conf = mock_pre.keypoints.conf
            assert xyn is not None and conf is not None
            pose8 = build_masked_pose_from_xyn(xyn[pick], conf[pick], min_kp_conf=min_kp)
            if not np.isnan(pose8).all():
                sc_stats["frames_pose_any_kp"] += 1
            session_kms.append(keypoint_mask(pose8))
            session_poses.append(pose8)

    t = len(session_poses)
    poses_full_arr = np.stack(session_poses, axis=0) if t else np.zeros((0, 8, 2), np.float32)
    km_arr = np.stack(session_kms, axis=0) if t else np.zeros((0, 8), bool)
    user_dir = clip_root / f"user_{uid}"
    valid = write_session_close_artifacts(user_dir, poses_full_arr, km_arr)
    valid_frames = int(valid.sum())
    sc_stats["max_iou_anchor_sample"] = round(max_iou_seen, 4)

    if has_sidecar and sc_stats["frames_pick_iou_ok"] == 0:
        logger.warning(
            "session_close clip %s: 0 picks IoU>=%.2f (dets=%s anchors=%s) — "
            "revisa remap native→prep o bbox_space=%s",
            clip_name,
            min_iou,
            sc_stats["frames_with_detections"],
            sc_stats["frames_anchor_nonnull"],
            sidecar_info.bbox_space,
        )

    user_export_meta = {
        "frame_policy": "session_subject_iou",
        "iamodel_subject_bbox_json": str(sidecar_path.resolve()),
        "iamodel_subject_bbox_sidecar_resolved": str(sidecar_path.resolve()),
        "iamodel_session_close_export": True,
        "session_id": sidecar_info.session_id,
        "bbox_space": sidecar_info.bbox_space,
        "bbox_coord_frame": sidecar_info.bbox_coord_frame,
        "coord_width": sidecar_info.coord_width,
        "coord_height": sidecar_info.coord_height,
        "bbox_norm_divisor": sidecar_info.bbox_norm_divisor,
        "session_close_min_iou": min_iou,
        "match_pre_tracker": True,
        "max_iou_anchor_sample": sc_stats["max_iou_anchor_sample"],
        "sidecar_loaded": bool(has_sidecar),
        "bbox_by_frame_len": sidecar_info.bbox_by_frame_len,
        "frames_anchor_nonnull": sc_stats["frames_anchor_nonnull"],
        "frames_with_detections": sc_stats["frames_with_detections"],
        "frames_pick_iou_ok": sc_stats["frames_pick_iou_ok"],
        "frames_pose_any_kp": sc_stats["frames_pose_any_kp"],
        "anchor_remap_c1_live_frames": sc_stats["anchor_remap_c1_live_frames"],
        "anchor_remap_geometric_frames": sc_stats["anchor_remap_geometric_frames"],
        "valid_mask_sum": valid_frames,
        "poses_full_shape": [int(t), 8, 2],
    }
    (user_dir / "meta.json").write_text(
        json.dumps(user_export_meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if cfg.save_clip_mp4:
        dest_mp4 = clip_root / "clip.mp4"
        if video_path.resolve() != dest_mp4.resolve():
            shutil.copy2(video_path, dest_mp4)

    cat_val = str(cat) if cat is not None else "0"
    ffprobe_t = timing.nb_read_frames
    meta_users = [
        {
            "track_id": uid,
            "valid_pct": round(100.0 * valid_frames / max(t, 1), 2),
            "valid_frames": valid_frames,
            "total_frames": t,
            "poses_full_T": t,
            "passes_filters": True,
            "session_close_export": True,
            "subject_bbox_json": str(Path(subject_json).resolve()),
        }
    ]
    meta = {
        "clip_name": clip_name,
        "fps": fps,
        "frame_count": t,
        "poses_full_T": t,
        "frame_policy": "session_subject_iou",
        "video_frame_count": frame_count,
        "ffprobe_nb_read_frames": ffprobe_t,
        "decode_backend_used": timing.decode_backend_used or pipe_cfg.decode_backend,
        "frame_count_matches_ffprobe": (
            ffprobe_t is not None and t == int(ffprobe_t)
        ),
        "cat": cat_val,
        "clip_duration": round(t / fps, 3) if fps > 0 else 0,
        "yolo_backend": "triton_tensorrt_plan",
        "pose_bundle_id": stack.bundle.get("pose_bundle_id", ""),
        "users": meta_users,
        "clip_video": "clip.mp4",
        "iamodel_extract_engine": "c1_like",
        "iamodel_decode_backend": pipe_cfg.decode_backend,
        "iamodel_preprocess_workers": pipe_cfg.preprocess_workers,
        "iamodel_session_close_export": True,
        "iamodel_subject_bbox_json": str(sidecar_path.resolve()),
        "iamodel_session_close_min_iou": min_iou,
        "iamodel_session_close_stats": sc_stats,
        "bbox_space": sidecar_info.bbox_space,
        "bbox_coord_frame": sidecar_info.bbox_coord_frame,
        "sidecar_anchor_nonnull": sidecar_info.anchor_nonnull,
        "session_close_match": sc_stats,
    }
    (clip_root / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (clip_root / "pose_bundle.json").write_text(
        json.dumps(stack.bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    t_infer = time.perf_counter() - t2
    timings = {
        "decode_sec": round(t_decode, 3),
        "preprocess_sec": round(t_prep, 3),
        "infer_track_export_sec": round(t_infer, 3),
        "total_sec": round(time.perf_counter() - t0, 3),
    }
    logger.info(
        "session_subject clip %s user_%d frames=%d valid=%d json=%s",
        clip_name,
        uid,
        t,
        valid_frames,
        subject_json,
    )
    result = ClipPoseResult(
        clip_name=clip_name,
        out_dir=clip_root,
        meta=meta,
        user_dirs=[user_dir],
    )
    return result, timings


def _infer_batched(stack: PoseExtractStack, tensors: list[np.ndarray]) -> list[np.ndarray]:
    from pose_core.postprocess import split_triton_batch_output

    bs = max(1, int(stack.extract_cfg.triton_batch_size))
    outputs: list[np.ndarray] = []
    for start in range(0, len(tensors), bs):
        chunk = tensors[start : start + bs]
        batch = np.stack(chunk, axis=0)
        out = stack.client.infer(batch)
        outputs.extend(split_triton_batch_output(out, len(chunk)))
    return outputs


def _none_tracker_pick(mock):
    if mock.boxes.xyxy.shape[0] == 0:
        return mock
    best = int(np.argmax(mock.boxes.conf))
    from pose_core.yolo_result import slice_mock_detection

    return slice_mock_detection(mock, best, track_id=0)


def _append_nan_row(acc: _UserAccum) -> None:
    acc.poses_full.append(NAN_POSE.copy())
    acc.keypoint_masks.append(EMPTY_KP_MASK.copy())
    acc.total += 1


def _process_frame_keep_all(
    users: dict[int, _UserAccum],
    mock,
    cfg,
    frame_index: int,
) -> None:
    """
    Una fila poses_full por frame decodificado (grid denso C1 / gold N2).
    Sin detección o keypoints ausentes → NaN; track ausente en frame → NaN.
    """
    from pose_core.export import build_masked_pose_from_xyn, frame_valid, keypoint_mask

    n_boxes = int(mock.boxes.xyxy.shape[0]) if mock.boxes.xyxy is not None else 0
    ids_raw = getattr(mock.boxes, "id", None)
    xyn = mock.keypoints.xyn
    conf = mock.keypoints.conf

    id_to_pose: dict[int, np.ndarray] = {}
    if ids_raw is not None and n_boxes > 0 and xyn is not None and conf is not None:
        ids = ids_raw.int().cpu().tolist() if hasattr(ids_raw, "cpu") else list(ids_raw)
        min_kp = float(getattr(cfg, "min_kp_conf", 0.5))
        for i, tid in enumerate(ids):
            tid = int(tid)
            id_to_pose[tid] = build_masked_pose_from_xyn(xyn[i], conf[i], min_kp_conf=min_kp)

    active_tids = set(users.keys()) | set(id_to_pose.keys())
    for tid in sorted(active_tids):
        if tid not in users:
            users[tid] = _UserAccum(poses_full=[], poses=[], keypoint_masks=[])
            for _ in range(frame_index):
                _append_nan_row(users[tid])
        acc = users[tid]
        while len(acc.poses_full) < frame_index:
            _append_nan_row(acc)
        if tid in id_to_pose:
            pose8 = id_to_pose[tid]
            km = keypoint_mask(pose8)
            acc.poses_full.append(pose8)
            acc.keypoint_masks.append(km)
            if frame_valid(pose8):
                acc.poses.append(pose8)
                acc.body_visible_cnt += 1
            acc.total += 1
        else:
            _append_nan_row(acc)


def _assert_keep_all_timeline(
    users: dict[int, _UserAccum], decoded_t: int, clip_name: str
) -> None:
    for tid, acc in users.items():
        t = len(acc.poses_full)
        if t != decoded_t:
            logger.warning(
                "clip %s user_%s: poses_full T=%d != decoded_T=%d tras infer",
                clip_name,
                tid,
                t,
                decoded_t,
            )


def _accumulate_frame(users: dict[int, _UserAccum], mock, cfg, *, frame_policy: str) -> None:
    from pose_core.export import build_masked_pose_from_xyn, frame_valid, keypoint_mask

    ids_raw = getattr(mock.boxes, "id", None)
    has_ids = ids_raw is not None and len(mock.boxes.xyxy) > 0
    if not has_ids:
        if frame_policy == "iamodel":
            return
        for acc in users.values():
            _append_nan_row(acc)
        return

    ids = ids_raw.int().cpu().tolist() if hasattr(ids_raw, "cpu") else list(ids_raw)
    xyn = mock.keypoints.xyn
    conf = mock.keypoints.conf
    if xyn is None or conf is None:
        if frame_policy == "iamodel":
            return
        for acc in users.values():
            _append_nan_row(acc)
        return

    for tid in list(users.keys()):
        if tid not in ids:
            if frame_policy == "iamodel":
                continue
            _append_nan_row(users[tid])

    min_kp = float(getattr(cfg, "min_kp_conf", 0.5))
    for i, tid in enumerate(ids):
        tid = int(tid)
        if tid not in users:
            users[tid] = _UserAccum(poses_full=[], poses=[], keypoint_masks=[])
        pose8 = build_masked_pose_from_xyn(xyn[i], conf[i], min_kp_conf=min_kp)
        km = keypoint_mask(pose8)
        acc = users[tid]
        acc.poses_full.append(pose8)
        acc.keypoint_masks.append(km)
        if frame_valid(pose8):
            acc.poses.append(pose8)
            acc.body_visible_cnt += 1
        acc.total += 1


def _passes_filters(acc: _UserAccum, total_frames: int) -> bool:
    from pose_core.export import BODY_VISIBLE_MIN_FRAMES, BODY_VISIBLE_MIN_RATIO

    if acc.body_visible_cnt < BODY_VISIBLE_MIN_FRAMES:
        return False
    if acc.body_visible_cnt / max(acc.total, 1) < BODY_VISIBLE_MIN_RATIO:
        return False
    if total_frames > 0 and acc.total < max(1, int(0.2 * total_frames)):
        return False
    return True
