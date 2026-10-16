"""Construcción de cola P0 (eval incumbent paralela)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gpu_scheduler import JobSpec


def _resolve_model_path(entry: Dict[str, Any]) -> Optional[Path]:
    raw = (entry.get("path") or "").strip()
    mid = str(entry.get("id", "")).strip()
    if not raw and mid:
        raw = (os.environ.get(f"CAMPAIGN2_MODEL_{mid}") or "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    return p if p.is_file() else None


def resolve_p0_models(cfg: Dict[str, Any]) -> List[Tuple[str, Path]]:
    p0 = cfg.get("p0_incumbent") or {}
    models_cfg = p0.get("models") or []
    resolved: List[Tuple[str, Path]] = []
    for m in models_cfg:
        mid = str(m.get("id", ""))
        p = _resolve_model_path(m)
        if p is not None:
            resolved.append((mid, p))
    return resolved


def build_p0_job_queue(
    cfg: Dict[str, Any],
    *,
    manifest_path: Path,
    data_roots: List[Path],
    run_id: str,
    pose_source: Optional[str] = None,
    id_suffix: str = "",
) -> Tuple[List[JobSpec], List[str]]:
    p0 = cfg.get("p0_incumbent") or {}
    if not p0.get("enabled", True):
        return [], ["p0_incumbent.enabled=false"]

    models = resolve_p0_models(cfg)
    warnings: List[str] = []
    if not models:
        warnings.append(
            "Sin rutas a modelo_41/modelo_48: config p0_incumbent.models[].path "
            "o env CAMPAIGN2_MODEL_41 / CAMPAIGN2_MODEL_48"
        )
        return [], warnings

    eval_split = str(p0.get("eval_split", "val_dev"))
    modes = p0.get("modes") or ["legacy_clip", "windows"]
    su_variants = p0.get("single_user_variants")
    if su_variants is None:
        su_variants = [False, True]
    thresholds = p0.get("frozen_thresholds") or {}
    batch_size = int(p0.get("batch_size", 64))
    temporal = cfg.get("temporal") or {}
    pose_source = str(pose_source or cfg.get("pose_source", "full"))
    id_suffix = str(id_suffix or "")
    robbery_class = int(cfg.get("robbery_class", 6))

    jobs: List[JobSpec] = []
    priority = 10
    for mid, mpath in models:
        thr = float(thresholds.get(str(mid), thresholds.get(mid, 0.8)))
        for mode in modes:
            mode_tag = "a" if mode == "legacy_clip" else "b"
            if mode not in ("legacy_clip", "windows"):
                warnings.append(f"Modo P0 ignorado: {mode}")
                continue
            for su in su_variants:
                su_flag = 1 if bool(su) else 0
                job_id = f"p0{mode_tag}_m{mid}{id_suffix}_su{su_flag}"
                params: Dict[str, Any] = {
                    "mode": mode,
                    "model_id": mid,
                    "model_path": str(mpath),
                    "manifest_path": str(manifest_path),
                    "data_roots": [str(r) for r in data_roots],
                    "eval_split": eval_split,
                    "single_user_only": bool(su),
                    "pose_source": pose_source,
                    "robbery_class": robbery_class,
                    "batch_size": batch_size,
                    "frozen_threshold": thr,
                    "temporal": temporal,
                    "run_id": run_id,
                }
                jobs.append(
                    JobSpec(job_id=job_id, job_type="p0_eval", params=params, priority=priority)
                )
                priority += 1
    return jobs, warnings
