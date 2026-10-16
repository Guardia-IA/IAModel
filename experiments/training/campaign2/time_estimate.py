"""Estimaciones de tiempo Campaign2 (secuencial y paralelo)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

TRAINING_DIR = Path(__file__).resolve().parent.parent
import sys

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from training_time_estimate import (  # noqa: E402
    cuda_available,
    estimate_single_experiment_seconds,
    fmt_duration,
)

# Heurística v2 TCN pequeña (~masked dilated width 64); cercana a dilated_tcn + algo de overhead.
ARCH_ALIASES = {
    "masked_tcn64": "dilated_tcn",
    "masked_multiscale_tcn_se": "dilated_tcn",
    "ctr_gcn_lite": "stgcn",
    "patchtst_pose_lite": "transformer",
    "gru64_pooled": "gru",
}


def count_windows_in_manifest(
    entries: List[Dict[str, Any]],
    *,
    split: str,
    window_seconds: float,
    stride_seconds: float,
    target_hz: float,
) -> int:
    n = 0
    for e in entries:
        if e.get("split") != split:
            continue
        frames = int(e.get("num_frames") or 0)
        duration = frames / max(target_hz, 1e-6)
        if duration <= window_seconds:
            n += 1
        else:
            n += max(1, int((duration - window_seconds) / max(stride_seconds, 1e-6)) + 1)
    return n


def _job_to_legacy_cfg(job: Dict[str, Any]) -> Dict[str, Any]:
    arch = str(job.get("arch", "masked_tcn64"))
    legacy_arch = ARCH_ALIASES.get(arch, arch)
    return {
        "arch": legacy_arch,
        "epochs": int(job.get("epochs", 100)),
        "batch_size": int(job.get("batch_size", 64)),
        "seq_len": int(job.get("seq_len", 96)),
        "done": False,
    }


def estimate_job_plan(
    jobs: List[Dict[str, Any]],
    *,
    train_rows: int,
    val_rows: int,
    calibration_factor: float = 1.0,
) -> Dict[str, Any]:
    per_job: List[Dict[str, Any]] = []
    total_gpu = 0.0
    total_cpu = 0.0
    for job in jobs:
        cfg = _job_to_legacy_cfg(job)
        est = estimate_single_experiment_seconds(cfg, train_rows=train_rows, val_rows=val_rows)
        if est is None:
            continue
        gpu_s = est["gpu_seconds"] * calibration_factor
        cpu_s = est["cpu_seconds"] * calibration_factor
        total_gpu += gpu_s
        total_cpu += cpu_s
        per_job.append(
            {
                "job_id": job.get("job_id", "?"),
                "arch": job.get("arch", cfg["arch"]),
                "epochs": cfg["epochs"],
                "batch_size": cfg["batch_size"],
                "seq_len": cfg["seq_len"],
                "seeds": int(job.get("seeds", 1)),
                "gpu_seconds_per_seed": gpu_s,
                "cpu_seconds_per_seed": cpu_s,
                "gpu_seconds_all_seeds": gpu_s * int(job.get("seeds", 1)),
            }
        )

    on_gpu = cuda_available()
    serial = total_gpu if on_gpu else total_cpu
    # Expand by seeds per job spec
    serial_with_seeds = sum(
        j["gpu_seconds_all_seeds"] if on_gpu else j["cpu_seconds_per_seed"] * j["seeds"]
        for j in per_job
    )

    return {
        "train_rows_per_epoch": train_rows,
        "val_rows_per_epoch": val_rows,
        "cuda_available": on_gpu,
        "calibration_factor": calibration_factor,
        "jobs": per_job,
        "serial_seconds": float(serial_with_seeds),
        "serial_gpu_seconds": float(sum(j["gpu_seconds_all_seeds"] for j in per_job)),
        "serial_cpu_seconds": float(sum(j["cpu_seconds_per_seed"] * j["seeds"] for j in per_job)),
    }


def parallel_wall_seconds(serial_seconds: float, num_jobs: int, max_parallel: int) -> float:
    if num_jobs <= 0 or serial_seconds <= 0:
        return 0.0
    slots = max(1, min(int(max_parallel), int(num_jobs)))
    # Cola: suma de duraciones / slots + 10% overhead scheduler/I/O
    return (serial_seconds / slots) * 1.10


def default_p1_job_plan(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    te = cfg.get("time_estimate") or {}
    if te.get("p1_job_matrix"):
        return list(te["p1_job_matrix"])
    seeds = cfg.get("seeds") or [17, 42, 103]
    temporal = cfg.get("temporal") or {}
    seq = int(temporal.get("seq_len", 96))
    return [
        {
            "job_id": "v2_tcn64_baseline",
            "arch": "masked_tcn64",
            "epochs": 100,
            "batch_size": 64,
            "seq_len": seq,
            "seeds": len(seeds),
        },
        {
            "job_id": "v2_gru64_pooled",
            "arch": "gru64_pooled",
            "epochs": 100,
            "batch_size": 64,
            "seq_len": seq,
            "seeds": len(seeds),
        },
    ]


def build_time_estimate_report(
    cfg: Dict[str, Any],
    *,
    manifest_entries: List[Dict[str, Any]],
    max_parallel_jobs: int,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    temporal = cfg.get("temporal") or {}
    te_cfg = cfg.get("time_estimate") or {}
    calib = float(te_cfg.get("legacy_calibration_factor", 7.0))
    te_cfg.get("legacy_calibration_factor_comment", "Oct 2026: preflight ~57m vs ~7h wall por celda")

    window_s = float(temporal.get("window_seconds", 8.0))
    stride_s = float(temporal.get("stride_seconds", 1.0))
    hz = float(temporal.get("target_hz", 12.0))

    train_rows = count_windows_in_manifest(
        manifest_entries, split="train", window_seconds=window_s, stride_seconds=stride_s, target_hz=hz
    )
    val_rows = count_windows_in_manifest(
        manifest_entries, split="val_dev", window_seconds=window_s, stride_seconds=stride_s, target_hz=hz
    )
    if train_rows == 0 and manifest_entries:
        # Fallback: 70% clips × 3 ventanas
        train_rows = max(1, int(len(manifest_entries) * 0.7 * 3))
        val_rows = max(1, int(len(manifest_entries) * 0.1 * 3))

    jobs = default_p1_job_plan(cfg)
    num_runs = sum(int(j.get("seeds", 1)) for j in jobs)
    plan = estimate_job_plan(jobs, train_rows=train_rows, val_rows=val_rows, calibration_factor=calib)
    parallel_s = parallel_wall_seconds(plan["serial_seconds"], num_runs, max_parallel_jobs)

    # Legacy reference: 20 models × 1 cell sequential calibrated
    legacy_20 = None
    try:
        from model_config import EXPERIMENTS  # noqa: E402
        from training_time_estimate import estimate_all_experiments  # noqa: E402

        ids = list(range(1, min(21, len(EXPERIMENTS) + 1)))
        leg = estimate_all_experiments(
            train_rows=max(train_rows, 16000),
            val_rows=max(val_rows, 2000),
            experiment_ids=ids,
        )
        leg_serial = leg["primary_total_seconds"] * calib
        legacy_20 = {
            "experiments": len(ids),
            "serial_seconds_calibrated": leg_serial,
            "parallel_seconds": parallel_wall_seconds(leg_serial, len(ids), max_parallel_jobs),
        }
    except Exception:
        legacy_20 = None

    return {
        "run_id": run_id,
        "max_parallel_jobs": max_parallel_jobs,
        "window_train_rows_per_epoch": train_rows,
        "window_val_rows_per_epoch": val_rows,
        "p1_planned_jobs": len(jobs),
        "p1_planned_training_runs": num_runs,
        "calibration_factor_applied": calib,
        "note": (
            "Heurística + factor calibración campaña Oct 2026 (estimador legacy subestima ~7×). "
            "Sprint 2 no implementado aún; esto es plan P1."
        ),
        "p1_serial_seconds": plan["serial_seconds"],
        "p1_parallel_seconds": parallel_s,
        "p1_serial_human": fmt_duration(plan["serial_seconds"]),
        "p1_parallel_human": fmt_duration(parallel_s),
        "per_job": plan["jobs"],
        "legacy_campaign_20_models_reference": legacy_20,
        "phase": str(cfg.get("phase", "sprint1_infra")),
    }
