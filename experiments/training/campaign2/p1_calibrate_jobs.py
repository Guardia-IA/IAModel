"""Cola calibración umbral T para checkpoints P1."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from gpu_scheduler import JobSpec

from manifest_paths import manifest_for_job_id
from p1_eval_jobs import discover_trained_checkpoints


def build_p1_calibrate_job_queue(
    cfg: Dict[str, Any],
    *,
    manifest_path: str,
    run_id: str,
    jobs_dir: Path,
) -> Tuple[List[JobSpec], List[str]]:
    warnings: List[str] = []
    ckpts = discover_trained_checkpoints(jobs_dir)
    if not ckpts:
        warnings.append("No hay best.pt — entrena P1 antes de p1-calibrate")
        return [], warnings

    temporal = cfg.get("temporal") or {}
    p1_train = dict(cfg.get("p1_train") or {})
    selection = dict(cfg.get("selection") or {})
    calib_split = str((cfg.get("p1_calibrate") or {}).get("split", "threshold_calib"))
    batch_size = int((cfg.get("p1_calibrate") or {}).get("batch_size", 64))

    jobs: List[JobSpec] = []
    for job_id, ckpt_path in ckpts:
        cid = f"calib_{job_id}"
        man = manifest_path
        mp = manifest_for_job_id(run_id, job_id)
        if mp is not None and mp.is_file():
            man = str(mp)
        params: Dict[str, Any] = {
            "source_job_id": job_id,
            "checkpoint_path": str(ckpt_path),
            "manifest_path": man,
            "calib_split": calib_split,
            "batch_size": batch_size,
            "temporal": temporal,
            "p1_train": p1_train,
            "selection": selection,
            "run_id": run_id,
        }
        jobs.append(JobSpec(job_id=cid, job_type="p1_calibrate", params=params, priority=5))
    return jobs, warnings


def write_calibrate_queue(run_id: str, jobs: List[JobSpec]) -> Path:
    from campaign2_paths import job_queue_path

    qpath = job_queue_path(run_id)
    payload = {
        "run_id": run_id,
        "phase": "sprint3_p1_calibrate",
        "purpose": "p1_calibrate",
        "jobs": [
            {"job_id": j.job_id, "job_type": j.job_type, "params": j.params, "priority": j.priority}
            for j in jobs
        ],
    }
    qpath.parent.mkdir(parents=True, exist_ok=True)
    with open(qpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return qpath
