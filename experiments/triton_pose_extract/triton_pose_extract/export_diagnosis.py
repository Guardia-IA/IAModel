"""Clasificación de export pose (alineado con handoff C1 / clip_extractor v1)."""
from __future__ import annotations

from typing import Any, Dict, Literal, Tuple

ExportStatus = Literal[
    "ok",
    "skip_no_export",
    "skip_no_tracks",
    "fail_infra",
]


def infer_export_status(
    meta: Dict[str, Any] | None,
    *,
    single_user_only: bool,
    validation_errors: list[str] | None = None,
) -> Tuple[ExportStatus, str]:
    """
    meta.json no incluye skip_reason en vendor v1; inferimos desde users/frame_count + config.
    Correlacionar siempre con logs pose_core (-v).
    """
    validation_errors = validation_errors or []
    if not meta:
        return "fail_infra", "sin meta.json"

    users = meta.get("users") or []
    if users:
        if validation_errors:
            return "fail_infra", ";".join(validation_errors)
        return "ok", ""

    frame_count = int(meta.get("frame_count") or 0)
    if frame_count <= 0:
        return "skip_no_tracks", "0 frames / sin tracks (decode o tracker)"

    if single_user_only:
        return (
            "skip_no_export",
            "sin npy: users[] por multi-track + single_user_only "
            f"(no es «sin persona»; ver log «single_user_only y N tracks»; {frame_count} frames)",
        )

    return (
        "skip_no_export",
        f"sin npy: users[] tras filtros BODY_VISIBLE/calidad ({frame_count} frames)",
    )


def is_infra_failure(status: ExportStatus, error: str | None) -> bool:
    if status == "fail_infra":
        return True
    if error in ("ffmpeg_falló",):
        return True
    if error and error.startswith("PermissionError"):
        return True
    return False
