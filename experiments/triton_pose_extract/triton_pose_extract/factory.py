"""Construye ClipExtractor (un cliente gRPC por worker)."""
from __future__ import annotations

from triton_pose_extract.extract_stack import PoseExtractStack, build_extract_stack
from triton_pose_extract.settings import ExtractSettings


def build_clip_extractor(settings: ExtractSettings):
    stack = build_extract_stack(settings)
    from pose_core.clip_extractor import ClipExtractor

    return ClipExtractor(
        stack.client,
        stack.preprocess,
        stack.tracker,
        stack.bundle,
        extract_cfg=stack.extract_cfg,
    )


def build_pose_stack(settings: ExtractSettings) -> PoseExtractStack:
    return build_extract_stack(settings)
