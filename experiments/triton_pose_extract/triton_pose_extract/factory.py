"""Construye ClipExtractor (un cliente gRPC por worker)."""
from __future__ import annotations

import json
from pathlib import Path

from triton_pose_extract.paths import ensure_pose_core_importable
from triton_pose_extract.settings import ExtractSettings


def build_clip_extractor(settings: ExtractSettings):
    ensure_pose_core_importable(settings.triton_pose_repo)
    from pose_core.clip_extractor import ClipExtractConfig, ClipExtractor
    from pose_core.config import load_pose_config
    from pose_core.preprocess import PreprocessConfig
    from pose_core.tracker import TrackerAdapter
    from pose_core.triton_client import TritonPoseClient

    cfg = load_pose_config(settings.config_path if settings.config_path.is_file() else None)
    triton = cfg.get("triton") or {}
    preprocess_cfg = cfg.get("preprocess") or {}
    clip_cfg_yaml = cfg.get("clip") or {}

    client = TritonPoseClient(
        url=settings.triton_url,
        model_name=settings.model_name,
        imgsz=int(triton.get("tracking_imgsz") or 640),
        input_fp16=settings.input_fp16,
        infer_timeout_seconds=float(settings.infer_timeout_seconds),
        use_system_shm=False,
    )
    client.connect()

    bundle_path = settings.bundle_json
    if bundle_path.is_file():
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    else:
        bundle = {"pose_bundle_id": cfg.get("pose_bundle_id", "unknown")}

    pp = PreprocessConfig(
        infer_max_width=int(preprocess_cfg.get("infer_frame_max_width") or 640),
        infer_max_height=int(preprocess_cfg.get("infer_frame_max_height") or 360),
        imgsz=int(triton.get("tracking_imgsz") or 640),
        input_fp16=settings.input_fp16,
        input_color=str(preprocess_cfg.get("input_color") or "bgr"),  # type: ignore[arg-type]
        model_color=str(preprocess_cfg.get("model_color") or "rgb"),  # type: ignore[arg-type]
    )
    extract_cfg = ClipExtractConfig(
        parse_conf=settings.parse_conf,
        triton_batch_size=settings.triton_batch_size,
        tracker_kind=settings.tracker,  # type: ignore[arg-type]
        single_user_only=settings.single_user_only,
        frame_policy=str(clip_cfg_yaml.get("frame_policy") or "keep_all"),  # type: ignore[arg-type]
    )
    tracker = TrackerAdapter(
        kind=extract_cfg.tracker_kind,
        config_path=clip_cfg_yaml.get("tracker_yaml"),
        frame_rate=15.0,
        imgsz=pp.imgsz,
        track_conf=float(clip_cfg_yaml.get("track_conf") or 0.22),
    )
    return ClipExtractor(client, pp, tracker, bundle, extract_cfg=extract_cfg)
