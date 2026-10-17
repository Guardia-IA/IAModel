"""Campaña matriz: full + filtered (y más) en una sola cola."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from gpu_scheduler import JobSpec
from p0_jobs import build_p0_job_queue
from p1_jobs import build_p1_job_queue


def matrix_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    m = cfg.get("campaign_matrix") or {}
    return {
        "enabled": bool(m.get("enabled", False)),
        "pose_sources": list(m.get("pose_sources") or ["full", "filtered"]),
        "run_p1_train": bool(m.get("run_p1_train", True)),
        "run_p0_legacy": bool(m.get("run_p0_legacy", True)),
        "p0_modes": list(m.get("p0_modes") or ["windows"]),
        "single_user_variants": list(m.get("single_user_variants") or [False]),
        "ensemble_pairs": list(m.get("ensemble_pairs") or []),
        "auto_pairs_top_legacy": bool(m.get("auto_pairs_from_incumbent_models", True)),
    }


def cache_dir_for_pose(layout_cache: Path, pose_source: str) -> Path:
    d = layout_cache / str(pose_source)
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_matrix_job_queue(
    cfg: Dict[str, Any],
    *,
    manifest_by_pose: Dict[str, Path],
    data_roots: List[Path],
    run_id: str,
    smoke: bool = False,
    debug: bool = False,
) -> Tuple[List[JobSpec], List[str]]:
    ms = matrix_settings(cfg)
    warnings: List[str] = []
    jobs: List[JobSpec] = []

    dbg = cfg.get("debug") or {}
    run_p0 = ms["run_p0_legacy"]
    if debug and dbg.get("run_p0_legacy") is False:
        run_p0 = False

    # P0 legacy: por pose_source y cada .pt configurado (opcional, solo comparación)
    if run_p0 and not smoke:
        p0_cfg = dict(cfg.get("p0_incumbent") or {})
        p0_cfg["modes"] = ms["p0_modes"]
        p0_cfg["single_user_variants"] = ms["single_user_variants"]
        cfg_p0 = dict(cfg)
        cfg_p0["p0_incumbent"] = p0_cfg
        for ps in ms["pose_sources"]:
            man = manifest_by_pose.get(ps)
            if not man or not man.is_file():
                warnings.append(f"P0: sin manifest para pose_source={ps}")
                continue
            jq, w = build_p0_job_queue(
                cfg_p0,
                manifest_path=man,
                data_roots=data_roots,
                run_id=run_id,
                pose_source=ps,
                id_suffix=f"_{ps}",
            )
            warnings.extend(w)
            jobs.extend(jq)

    # P1 train_v2: duplicar matriz por pose_source
    if ms["run_p1_train"]:
        for ps in ms["pose_sources"]:
            man = manifest_by_pose.get(ps)
            if not man:
                continue
            jq, w = build_p1_job_queue(
                cfg,
                manifest_path=str(man),
                run_id=run_id,
                smoke=smoke,
                debug=debug,
                id_suffix=f"_{ps}",
                pose_source=ps,
            )
            warnings.extend(w)
            jobs.extend(jq)

    return jobs, warnings
