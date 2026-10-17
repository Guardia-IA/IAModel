"""Configuración de extracción (JSON + env + pose.yaml del vendor)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from triton_pose_extract.paths import default_triton_pose_repo, ensure_pose_core_importable


@dataclass
class ExtractSettings:
    triton_pose_repo: Path
    triton_url: str
    http_health_base: str
    model_name: str
    config_path: Path
    bundle_json: Path
    infer_timeout_seconds: float
    triton_batch_size: int
    tracker: str
    parse_conf: float
    single_user_only: bool
    parallel_clips: int | None
    max_batch_size: int
    gpu_instance_count: int
    input_fp16: bool

    @classmethod
    def load(
        cls,
        config_json: Path | None = None,
        *,
        triton_url: str | None = None,
        parallel_clips: int | None = None,
    ) -> "ExtractSettings":
        pkg = Path(__file__).resolve().parent.parent
        cfg_path = config_json or pkg / "triton_pose_extract_config.json"
        raw: dict[str, Any] = {}
        if cfg_path.is_file():
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))

        repo = default_triton_pose_repo()
        if raw.get("kanvis_triton_pose_root"):
            repo = Path(str(raw["kanvis_triton_pose_root"])).expanduser().resolve()
        if repo is None:
            raise RuntimeError("kanvis-triton-pose no encontrado")
        ensure_pose_core_importable(repo)

        from pose_core.config import default_config_path, load_pose_config

        pose_yaml = Path(raw.get("pose_config") or default_config_path())
        if not pose_yaml.is_file():
            pose_yaml = repo / "config" / "pose.yaml"
        pose = load_pose_config(pose_yaml if pose_yaml.is_file() else None)
        triton = pose.get("triton") or {}
        clip = pose.get("clip") or {}

        url = (
            triton_url
            or os.environ.get("KANVIS_TRITON_URL", "").strip()
            or str(raw.get("triton_url") or triton.get("triton_url") or "kanvis-pilot.ai:47811")
        ).replace("grpc://", "")

        http_base = str(
            raw.get("http_health_base")
            or os.environ.get("KANVIS_TRITON_HTTP", "").strip()
            or _grpc_to_http_health(url)
        )

        bundle = Path(
            raw.get("bundle_json")
            or repo / "triton" / "model-repo" / "yolo26s-pose" / "1" / "pose_bundle.json"
        )

        return cls(
            triton_pose_repo=repo,
            triton_url=url,
            http_health_base=http_base.rstrip("/"),
            model_name=str(raw.get("model_name") or triton.get("model_name") or "yolo26s-pose"),
            config_path=pose_yaml,
            bundle_json=bundle,
            infer_timeout_seconds=float(
                raw.get("infer_timeout_seconds") or os.environ.get("KANVIS_TRITON_INFER_TIMEOUT") or 300.0
            ),
            triton_batch_size=int(raw.get("triton_batch_size") or clip.get("triton_batch_size") or 8),
            tracker=str(raw.get("tracker") or clip.get("tracker") or "bytetrack"),
            parse_conf=float(raw.get("parse_conf") or clip.get("parse_conf") or 0.18),
            single_user_only=bool(raw.get("single_user_only", False)),
            parallel_clips=parallel_clips if parallel_clips is not None else raw.get("parallel_clips"),
            max_batch_size=int(triton.get("max_batch_size") or 32),
            gpu_instance_count=int(triton.get("gpu_instance_count") or 2),
            input_fp16=bool(triton.get("input_fp16", True)),
        )


def _grpc_to_http_health(grpc_url: str) -> str:
    """Host piloto: gRPC 47811 → HTTP 47810; Docker 8001 → 8000."""
    host = grpc_url.split(":")[0] if ":" in grpc_url else grpc_url
    port = 47810
    if ":8001" in grpc_url or grpc_url.endswith("8001"):
        port = 8000
    elif ":47811" in grpc_url:
        port = 47810
    return f"http://{host}:{port}"
