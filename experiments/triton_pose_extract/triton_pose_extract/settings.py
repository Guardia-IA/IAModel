"""Configuración de extracción (JSON + env + pose.yaml del vendor)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from triton_pose_extract.paths import default_triton_pose_repo


@dataclass
class ExtractSettings:
    triton_pose_repo: Optional[Path]
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
    use_system_shm: bool

    @classmethod
    def load(
        cls,
        config_json: Path | None = None,
        *,
        triton_url: str | None = None,
        parallel_clips: int | None = None,
        require_pose_repo: bool = False,
    ) -> "ExtractSettings":
        """
        require_pose_repo=False → check-triton / preflight HTTP (solo URL).
        require_pose_repo=True  → extract (necesita pose_core en disco).
        """
        pkg = Path(__file__).resolve().parent.parent
        cfg_path = config_json or pkg / "triton_pose_extract_config.json"
        raw: dict[str, Any] = {}
        if cfg_path.is_file():
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))

        repo: Optional[Path] = default_triton_pose_repo()
        if raw.get("kanvis_triton_pose_root"):
            p = Path(str(raw["kanvis_triton_pose_root"])).expanduser().resolve()
            if p.is_dir():
                repo = p

        triton_cfg: dict[str, Any] = dict(raw.get("triton") or {})
        clip_cfg: dict[str, Any] = dict(raw.get("clip") or {})
        pose_yaml = Path(str(raw.get("pose_config") or "")).expanduser() if raw.get("pose_config") else Path()

        if repo is not None and not pose_yaml.is_file():
            pose_yaml = repo / "config" / "pose.yaml"
        if pose_yaml.is_file():
            try:
                import yaml

                data = yaml.safe_load(pose_yaml.read_text(encoding="utf-8")) or {}
                if isinstance(data, dict):
                    triton_cfg = {**data.get("triton", {}), **triton_cfg}
                    clip_cfg = {**data.get("clip", {}), **clip_cfg}
            except Exception:
                pass

        if require_pose_repo and repo is None:
            raise RuntimeError(
                "kanvis-triton-pose no encontrado en el host. "
                "El contenedor Docker triton-pose NO incluye pose_core para Python. "
                "Clona el repo y exporta:\n"
                "  export KANVIS_TRITON_POSE_ROOT=/home/angel/kanvis-triton-pose\n"
                "(check-triton funciona sin repo; extract-csv sí lo necesita)."
            )

        filt_raw = raw.get("filters") or {}
        su_raw = filt_raw.get("single_user_only", raw.get("single_user_only", True))
        if isinstance(su_raw, dict):
            single_user_only = bool(su_raw.get("enabled", True))
        else:
            single_user_only = bool(su_raw)

        url = (
            triton_url
            or os.environ.get("KANVIS_TRITON_URL", "").strip()
            or str(raw.get("triton_url") or triton_cfg.get("triton_url") or "127.0.0.1:47811")
        ).replace("grpc://", "")

        http_base = str(
            raw.get("http_health_base")
            or os.environ.get("KANVIS_TRITON_HTTP", "").strip()
            or _grpc_to_http_health(url)
        )

        bundle = Path(str(raw.get("bundle_json") or "")).expanduser() if raw.get("bundle_json") else Path()
        if repo and not bundle.is_file():
            bundle = repo / "triton" / "model-repo" / "yolo26s-pose" / "1" / "pose_bundle.json"
        if not bundle.is_file():
            bundle = pkg / "pose_bundle.json"

        return cls(
            triton_pose_repo=repo,
            triton_url=url,
            http_health_base=http_base.rstrip("/"),
            model_name=str(raw.get("model_name") or triton_cfg.get("model_name") or "yolo26s-pose"),
            config_path=pose_yaml if pose_yaml.is_file() else (repo / "config" / "pose.yaml" if repo else pkg / "pose.yaml"),
            bundle_json=bundle,
            infer_timeout_seconds=float(
                raw.get("infer_timeout_seconds") or os.environ.get("KANVIS_TRITON_INFER_TIMEOUT") or 300.0
            ),
            triton_batch_size=int(raw.get("triton_batch_size") or clip_cfg.get("triton_batch_size") or 8),
            tracker=str(raw.get("tracker") or clip_cfg.get("tracker") or "bytetrack"),
            parse_conf=float(raw.get("parse_conf") or clip_cfg.get("parse_conf") or 0.18),
            single_user_only=single_user_only,
            parallel_clips=parallel_clips if parallel_clips is not None else raw.get("parallel_clips"),
            max_batch_size=int(triton_cfg.get("max_batch_size") or 32),
            gpu_instance_count=int(triton_cfg.get("gpu_instance_count") or 2),
            input_fp16=bool(triton_cfg.get("input_fp16", True)),
            use_system_shm=_resolve_use_system_shm(raw, triton_cfg),
        )


def _resolve_use_system_shm(raw: dict, triton_cfg: dict) -> bool:
    env = os.environ.get("KANVIS_POSE_TRITON_SYSTEM_SHM_ENABLED", "").strip().lower()
    if env in ("1", "true", "yes"):
        return True
    if env in ("0", "false", "no"):
        return False
    if "use_system_shm" in triton_cfg:
        return bool(triton_cfg.get("use_system_shm"))
    return bool(raw.get("use_system_shm", False))


def _grpc_to_http_health(grpc_url: str) -> str:
    """Host local Docker: gRPC 47811 → HTTP 47810."""
    if "://" in grpc_url:
        grpc_url = grpc_url.split("://", 1)[-1]
    host = grpc_url.rsplit(":", 1)[0] if ":" in grpc_url else grpc_url
    port_part = grpc_url.rsplit(":", 1)[-1] if ":" in grpc_url else "47811"
    if port_part in ("8001", "47811"):
        http_port = 8000 if port_part == "8001" else 47810
    else:
        http_port = 47810
    return f"http://{host}:{http_port}"
