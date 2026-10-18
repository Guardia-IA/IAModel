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
) -> tuple[Any, dict[str, float]]:
    from pose_core.clip_extractor import ClipPoseResult
    from pose_core.export import (
        PersonRows,
        build_masked_pose_from_xyn,
        frame_valid,
        keypoint_mask,
        write_person_artifacts,
    )
    from pose_core.postprocess import decode_end2end, split_triton_batch_output
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
    )
    t_decode = time.perf_counter() - t0

    fps = timing.fps if timing.fps > 0 else DEFAULT_FPS
    frame_count = timing.frame_count or len(frames)
    cfg = stack.extract_cfg

    def _prep_one(frame: np.ndarray):
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
    for prep, lb_meta, raw_out in zip(prepared_list, meta_list, raw_batches, strict=True):
        dets = decode_end2end(raw_out, lb_meta, stack.decode_conf)
        mock = detections_to_mock(dets, orig_shape=(lb_meta.orig_h, lb_meta.orig_w))
        if cfg.tracker_kind == "none":
            mock = _none_tracker_pick(mock)
        else:
            mock = stack.tracker.update(mock, prep)
        mock = ensure_track_ids(mock)
        _accumulate_frame(users, mock, cfg, frame_policy=cfg.frame_policy)

    if cfg.single_user_only and len(users) > 1:
        logger.info("skip clip %s: single_user_only y %d tracks", clip_name, len(users))
        users = {}

    user_dirs: list[Path] = []
    meta_users: list[dict] = []
    filtered = 0
    for tid, acc in sorted(users.items()):
        if not _passes_filters(acc, frame_count):
            filtered += 1
            continue
        user_dir = clip_root / f"user_{tid}"
        write_person_artifacts(
            user_dir,
            PersonRows(poses_full=acc.poses_full, poses=acc.poses, keypoint_masks=acc.keypoint_masks),
        )
        user_dirs.append(user_dir)
        valid_frames = sum(1 for p in acc.poses_full if frame_valid(p, min_visible=5))
        meta_users.append(
            {
                "track_id": int(tid),
                "valid_pct": round(100.0 * valid_frames / max(acc.total, 1), 2),
                "valid_frames": valid_frames,
                "total_frames": acc.total,
                "passes_filters": True,
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
    meta = {
        "clip_name": clip_name,
        "fps": fps,
        "frame_count": len(prepared_list),
        "video_frame_count": frame_count,
        "cat": cat_val,
        "clip_duration": round(len(prepared_list) / fps, 3) if fps > 0 else 0,
        "yolo_backend": "triton_tensorrt_plan",
        "pose_bundle_id": stack.bundle.get("pose_bundle_id", ""),
        "users": meta_users,
        "clip_video": "clip.mp4",
        "iamodel_extract_engine": "c1_like",
        "iamodel_decode_backend": pipe_cfg.decode_backend,
        "iamodel_preprocess_workers": pipe_cfg.preprocess_workers,
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


def _accumulate_frame(users: dict[int, _UserAccum], mock, cfg, *, frame_policy: str) -> None:
    from pose_core.export import build_masked_pose_from_xyn, frame_valid, keypoint_mask

    ids_raw = getattr(mock.boxes, "id", None)
    has_ids = ids_raw is not None and len(mock.boxes.xyxy) > 0
    if not has_ids:
        if frame_policy == "iamodel":
            return
        for acc in users.values():
            acc.poses_full.append(NAN_POSE.copy())
            acc.keypoint_masks.append(EMPTY_KP_MASK.copy())
            acc.total += 1
        return

    ids = ids_raw.int().cpu().tolist() if hasattr(ids_raw, "cpu") else list(ids_raw)
    xyn = mock.keypoints.xyn
    conf = mock.keypoints.conf
    if xyn is None or conf is None:
        return

    for tid in list(users.keys()):
        if tid not in ids:
            if frame_policy == "iamodel":
                continue
            users[tid].poses_full.append(NAN_POSE.copy())
            users[tid].keypoint_masks.append(EMPTY_KP_MASK.copy())
            users[tid].total += 1

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
