#!/usr/bin/env python3
"""Ejecuta un job Campaign2 (subprocess aislado)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict


def _job_smoke(params: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    duration = float(params.get("sleep_seconds", 1.0))
    time.sleep(max(0.0, duration))
    gpu_info: Dict[str, Any] = {"cuda_available": False}
    try:
        import torch

        gpu_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            gpu_info["device_name"] = torch.cuda.get_device_name(0)
            x = torch.randn(4, 62, 96, device="cuda")
            y = x.mean()
            gpu_info["smoke_tensor_mean"] = float(y.item())
    except Exception as exc:
        gpu_info["error"] = str(exc)
    return {"job_type": "smoke", "gpu": gpu_info}


def _job_cache_touch(params: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    manifest = params.get("manifest_path")
    if not manifest or not Path(manifest).is_file():
        raise FileNotFoundError(f"manifest_path missing: {manifest}")
    with open(manifest, "r", encoding="utf-8") as f:
        man = json.load(f)
    import numpy as np

    poses_path = Path(man["poses_mmap"])
    mm = np.memmap(poses_path, dtype=np.float32, mode="r", shape=(man["total_frames"], man["num_joints"], 2))
    sample = float(mm[0, 0, 0]) if man["total_frames"] else 0.0
    return {"job_type": "cache_touch", "total_frames": man["total_frames"], "sample": sample}


def _job_smoke_train(params: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """Mini entrenamiento Conv1d para validar VRAM en worker real."""
    import torch
    import torch.nn as nn

    if not torch.cuda.is_available():
        return {"job_type": "smoke_train", "skipped": "no_cuda"}

    batch = int(params.get("batch_size", 32))
    seq_len = int(params.get("seq_len", 96))
    feat = int(params.get("feature_dim", 62))
    steps = int(params.get("steps", 5))

    dev = torch.device("cuda:0")
    model = nn.Sequential(
        nn.Conv1d(feat, 64, 3, padding=1),
        nn.GELU(),
        nn.Conv1d(64, 64, 3, padding=1, dilation=2),
        nn.GELU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(64, 1),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(batch, feat, seq_len, device=dev)
    y = torch.randint(0, 2, (batch, 1), device=dev, dtype=torch.float32)
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        opt.step()
    wall = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)
    ckpt = output_dir / "smoke_model.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt)
    return {
        "job_type": "smoke_train",
        "steps": steps,
        "wall_seconds": wall,
        "peak_vram_mb": round(float(peak_mb), 2),
        "checkpoint": str(ckpt),
    }


def _job_p0_eval(params: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    from p0_eval import run_p0_eval

    return run_p0_eval(params, output_dir)


HANDLERS = {
    "smoke": _job_smoke,
    "cache_touch": _job_cache_touch,
    "smoke_train": _job_smoke_train,
    "p0_eval": _job_p0_eval,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    args = ap.parse_args()

    spec_path = Path(args.spec)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    job_type = spec.get("job_type", "smoke")
    params = spec.get("params") or {}
    handler = HANDLERS.get(job_type)
    if handler is None:
        raise SystemExit(f"Unknown job_type: {job_type}")

    t0 = time.perf_counter()
    try:
        result = handler(params, output_dir)
        status = "ok"
        err = None
    except Exception as exc:
        result = {"job_type": job_type, "error": str(exc)}
        status = "failed"
        err = str(exc)

    payload = {
        "job_id": spec.get("job_id"),
        "run_id": spec.get("run_id"),
        "status": status,
        "wall_seconds": time.perf_counter() - t0,
        "result": result,
        "error": err,
    }
    with open(output_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
