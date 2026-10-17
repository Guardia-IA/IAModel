"""Cola de evaluación P1 (checkpoints best.pt)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from gpu_scheduler import JobSpec
from manifest_paths import manifest_for_job_id


def discover_trained_checkpoints(jobs_dir: Path) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    if not jobs_dir.is_dir():
        return out
    for d in sorted(jobs_dir.iterdir()):
        if not d.is_dir():
            continue
        ckpt = d / "best.pt"
        if ckpt.is_file():
            out.append((d.name, ckpt))
    return out


def build_p1_eval_job_queue(
    cfg: Dict[str, Any],
    *,
    manifest_path: str,
    run_id: str,
    jobs_dir: Path,
    only_job_ids: List[str] | None = None,
    use_per_job_manifest: bool = True,
) -> Tuple[List[JobSpec], List[str]]:
    warnings: List[str] = []
    ckpts = discover_trained_checkpoints(jobs_dir)
    if only_job_ids:
        allow = set(only_job_ids)
        ckpts = [(j, p) for j, p in ckpts if j in allow]
    if not ckpts:
        warnings.append("No hay best.pt — ejecuta train P1 antes de p1-eval")
        return [], warnings

    temporal = cfg.get("temporal") or {}
    p1_train = cfg.get("p1_train") or {}
    threshold = float((cfg.get("p1_eval") or {}).get("threshold", 0.5))
    eval_split = str((cfg.get("p1_eval") or {}).get("eval_split", "val_dev"))
    batch_size = int((cfg.get("p1_eval") or {}).get("batch_size", 64))

    jobs: List[JobSpec] = []
    for job_id, ckpt_path in ckpts:
        eid = f"eval_{job_id}"
        thr_path = ckpt_path.parent / "threshold.json"
        thr = threshold
        if thr_path.is_file():
            with open(thr_path, "r", encoding="utf-8") as f:
                thr = float(json.load(f).get("threshold", threshold))
        man = manifest_path
        if use_per_job_manifest:
            mp = manifest_for_job_id(run_id, job_id)
            if mp is not None and mp.is_file():
                man = str(mp)
        params: Dict[str, Any] = {
            "source_job_id": job_id,
            "checkpoint_path": str(ckpt_path),
            "manifest_path": man,
            "eval_split": eval_split,
            "threshold": thr,
            "batch_size": batch_size,
            "temporal": temporal,
            "p1_train": p1_train,
            "run_id": run_id,
        }
        jobs.append(JobSpec(job_id=eid, job_type="p1_eval", params=params, priority=5))
    return jobs, warnings


def write_eval_queue(run_id: str, jobs: List[JobSpec], purpose: str = "p1_eval") -> Path:
    from campaign2_paths import job_queue_path

    qpath = job_queue_path(run_id)
    payload = {
        "run_id": run_id,
        "phase": "sprint3_p1_eval",
        "purpose": purpose,
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
