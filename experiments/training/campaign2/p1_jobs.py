"""Cola de entrenamiento P1 (train_v2)."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from gpu_scheduler import JobSpec


def _seeds_for_job(job: Dict[str, Any], default_seeds: List[int]) -> List[int]:
    if isinstance(job.get("seed_list"), list):
        return [int(s) for s in job["seed_list"]]
    raw = job.get("seeds")
    if isinstance(raw, list):
        return [int(s) for s in raw]
    if isinstance(raw, int):
        if raw <= 10:
            return default_seeds[: max(1, raw)]
        return [raw]
    return list(default_seeds)


def build_p1_job_queue(
    cfg: Dict[str, Any],
    *,
    manifest_path: str,
    run_id: str,
    smoke: bool = False,
    debug: bool = False,
    id_suffix: str = "",
    pose_source: str = "full",
) -> Tuple[List[JobSpec], List[str]]:
    p1_phase = (cfg.get("phases") or {}).get("p1_train_v2") or {}
    if not p1_phase.get("enabled", False) and not smoke and not debug:
        return [], ["phases.p1_train_v2.enabled=false"]

    matrix = list((cfg.get("time_estimate") or {}).get("p1_job_matrix") or [])
    if debug:
        matrix = list((cfg.get("debug") or {}).get("p1_job_matrix") or matrix[:1])
    if smoke:
        matrix = [
            {
                "job_id": "p1_smoke",
                "arch": "masked_tcn64",
                "epochs": 2,
                "batch_size": 32,
                "seeds": 1,
            }
        ]

    default_seeds = [int(s) for s in (cfg.get("seeds") or [42])]
    temporal = cfg.get("temporal") or {}
    p1_train = dict(cfg.get("p1_train") or {})
    if debug:
        dbg = cfg.get("debug") or {}
        p1_train["max_windows_per_clip"] = int(dbg.get("max_windows_per_clip", 8))
        p1_train["early_stop_patience"] = int(dbg.get("early_stop_patience", 3))
    robbery_class = int(cfg.get("robbery_class", 6))
    warnings: List[str] = []
    jobs: List[JobSpec] = []

    for job in matrix:
        job_id = str(job.get("job_id", "v2_job"))
        seeds = _seeds_for_job(job, default_seeds)
        for seed in seeds:
            sid = f"{job_id}{id_suffix}_s{seed}"
            params: Dict[str, Any] = {
                "job_id": sid,
                "run_id": run_id,
                "manifest_path": manifest_path,
                "pose_source": str(pose_source),
                "arch": str(job.get("arch", "masked_tcn64")),
                "epochs": int(job.get("epochs", 100)),
                "batch_size": int(job.get("batch_size", 64)),
                "seq_len": int(job.get("seq_len", temporal.get("seq_len", 96))),
                "hidden_dim": int(job.get("hidden_dim", 64)),
                "num_layers": int(job.get("num_layers", 4)),
                "dropout": float(job.get("dropout", 0.15)),
                "seed": seed,
                "temporal": temporal,
                "p1_train": p1_train,
                "robbery_class": robbery_class,
                "num_workers": 0,
            }
            jobs.append(JobSpec(job_id=sid, job_type="p1_train", params=params, priority=20))

    if not jobs:
        warnings.append("p1_job_matrix vacía")
    return jobs, warnings
