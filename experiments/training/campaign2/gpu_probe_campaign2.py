#!/usr/bin/env python3
"""Estima VRAM por job y recomienda max_parallel_jobs (Sprint 1)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

CAMPAIGN2_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN2_DIR.parent
for p in (TRAINING_DIR, CAMPAIGN2_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from campaign2_paths import (  # noqa: E402
    load_campaign2_config,
    parallel_recommendation_path,
    probe_report_path,
    read_current_run_id,
)


def _probe_torch(
    *,
    batch_size: int,
    seq_len: int,
    feature_dim: int,
    device_index: int,
) -> Dict[str, Any]:
    import torch
    import torch.nn as nn

    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "message": "CUDA no disponible; max_parallel_jobs=1 (CPU)",
            "vram_total_mb": 0,
            "vram_per_job_mb": 0,
            "recommended_max_parallel_jobs": 1,
        }

    dev = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(dev)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)

    model = nn.Sequential(
        nn.Conv1d(feature_dim, 64, kernel_size=3, padding=1),
        nn.GELU(),
        nn.Conv1d(64, 64, kernel_size=3, padding=1, dilation=2),
        nn.GELU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(64, 1),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(batch_size, feature_dim, seq_len, device=dev, dtype=torch.float32)
    y = torch.randint(0, 2, (batch_size, 1), device=dev, dtype=torch.float32)

    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()

    peak_bytes = torch.cuda.max_memory_allocated(dev)
    total_bytes = torch.cuda.get_device_properties(dev).total_memory
    reserve_mb = 2048
    per_job_mb = peak_bytes / (1024 * 1024)
    usable_mb = total_bytes / (1024 * 1024) - reserve_mb
    # Overhead CUDA por proceso (~1 GB) estimado conservador
    process_overhead_mb = 1024
    effective_per_slot = max(per_job_mb + process_overhead_mb, process_overhead_mb)
    recommended = max(1, int(usable_mb // effective_per_slot))

    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(dev),
        "device_index": device_index,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "feature_dim": feature_dim,
        "vram_total_mb": round(total_bytes / (1024 * 1024), 1),
        "vram_peak_forward_backward_mb": round(per_job_mb, 1),
        "estimated_process_overhead_mb": process_overhead_mb,
        "vram_reserve_mb": reserve_mb,
        "recommended_max_parallel_jobs": recommended,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="GPU probe Campaign2")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--run-id", type=str, default=None)
    args = ap.parse_args()

    cfg = load_campaign2_config(Path(args.config) if args.config else None)
    par = cfg.get("parallelism") or {}
    run_id = args.run_id or read_current_run_id()
    if not run_id:
        print("ERROR: define RUN_ID o ejecuta run_campaign2.sh new-run", file=sys.stderr)
        return 2

    report = _probe_torch(
        batch_size=int(par.get("probe_batch_size", 64)),
        seq_len=int(par.get("probe_seq_len", 96)),
        feature_dim=int(par.get("probe_feature_dim", 62)),
        device_index=int(par.get("cuda_device_index", 0)),
    )

    config_cap = int(par.get("max_parallel_jobs", 8))
    if report.get("cuda_available"):
        rec = int(report["recommended_max_parallel_jobs"])
        chosen = min(config_cap, rec) if par.get("auto_tune_from_probe", True) else config_cap
    else:
        chosen = 1

    out = {
        "run_id": run_id,
        "probe": report,
        "config_max_parallel_jobs": config_cap,
        "chosen_max_parallel_jobs": chosen,
    }

    probe_path = probe_report_path(run_id)
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    with open(probe_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")

    rec_path = parallel_recommendation_path(run_id)
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "max_parallel_jobs": chosen,
                "source": "gpu_probe_campaign2.py",
            },
            f,
            indent=2,
        )
        f.write("\n")

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
