"""Paralelismo window_campaign: GPUs, VRAM y CPU."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple


def _parse_device_ids(par: Dict[str, Any]) -> Optional[List[str]]:
    raw = par.get("cuda_device_ids")
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return [str(int(x)) for x in raw]
    return [s.strip() for s in str(raw).split(",") if s.strip()]


def _probe_vram_slots(
    device_index: int,
    *,
    batch_size: int,
    seq_len: int,
    feature_dim: int,
    reserve_mb: float,
    process_overhead_mb: float,
) -> Dict[str, Any]:
    import torch
    import torch.nn as nn

    dev = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(dev)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)

    model = nn.Sequential(
        nn.Conv1d(feature_dim, 96, kernel_size=3, padding=1),
        nn.GELU(),
        nn.Conv1d(96, 96, kernel_size=3, padding=2, dilation=2),
        nn.GELU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(96, 1),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(batch_size, feature_dim, seq_len, device=dev)
    y = torch.randint(0, 2, (batch_size, 1), device=dev, dtype=torch.float32)

    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()

    peak = torch.cuda.max_memory_allocated(dev)
    total = torch.cuda.get_device_properties(dev).total_memory
    per_job_mb = peak / (1024 * 1024)
    total_mb = total / (1024 * 1024)
    usable = max(0.0, total_mb - reserve_mb)
    slot_mb = max(per_job_mb + process_overhead_mb, process_overhead_mb)
    slots = max(1, int(usable // slot_mb))
    return {
        "device_index": device_index,
        "device_name": torch.cuda.get_device_name(dev),
        "vram_total_mb": round(total_mb, 1),
        "vram_peak_mb": round(per_job_mb, 1),
        "slots": slots,
    }


def _heuristic_slots(
    device_index: int,
    *,
    est_gb_per_job: float,
    reserve_gb: float,
) -> Dict[str, Any]:
    import torch

    props = torch.cuda.get_device_properties(device_index)
    total_gb = props.total_memory / (1024**3)
    usable = max(0.0, total_gb - reserve_gb)
    slots = max(1, int(usable // max(est_gb_per_job, 0.5)))
    return {
        "device_index": device_index,
        "device_name": props.name,
        "vram_total_mb": round(total_gb * 1024, 1),
        "slots": slots,
        "heuristic": True,
    }


def resolve_parallelism(
    cfg: Dict[str, Any],
    *,
    override_parallel: Optional[int] = None,
) -> Tuple[int, List[str], int, Dict[str, Any]]:
    """
    Devuelve (max_parallel_jobs, lista GPU por job, dataloader_workers, informe).
    Lista GPU: repite índice si hay >1 job en la misma tarjeta (round-robin en el trainer).
    """
    par = cfg.get("parallelism") or {}
    train_def = cfg.get("train_defaults") or {}
    cap = max(1, int(par.get("max_parallel_jobs_cap", 8)))
    auto = bool(par.get("auto", True))
    report: Dict[str, Any] = {"auto": auto, "cap": cap}

    manual_ids = _parse_device_ids(par)
    cpus = os.cpu_count() or 8

    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
    except ImportError:
        cuda_ok = False
        torch = None  # type: ignore

    if override_parallel is not None:
        parallel = max(1, min(cap, int(override_parallel)))
        gpus = manual_ids or (["0"] if cuda_ok else ["0"])
        dl = _dataloader_workers(par, cpus, parallel)
        report.update({"source": "cli_override", "parallel": parallel, "gpus": gpus})
        return parallel, _expand_gpu_list(gpus, parallel), dl, report

    if not cuda_ok:
        parallel = 1
        dl = _dataloader_workers(par, cpus, parallel)
        report.update(
            {
                "source": "cpu_only",
                "parallel": parallel,
                "message": "Sin CUDA → 1 job en CPU",
            }
        )
        return parallel, ["0"], dl, report

    n_dev = torch.cuda.device_count()
    if manual_ids:
        device_indices = []
        for s in manual_ids:
            try:
                device_indices.append(int(s))
            except ValueError:
                continue
        if not device_indices:
            device_indices = list(range(n_dev))
    else:
        device_indices = list(range(n_dev))

    batch_size = int(train_def.get("batch_size", 32))
    seq_len = int(cfg.get("window_len", 96))
    feature_dim = int(par.get("probe_feature_dim", 68))
    reserve_mb = float(par.get("vram_reserve_mb", 2048))
    overhead_mb = float(par.get("process_overhead_mb", 1024))
    reserve_gb = float(par.get("vram_reserve_gb", reserve_mb / 1024))
    est_gb = float(par.get("estimated_vram_gb_per_job", 3.5))
    use_probe = bool(par.get("vram_probe", True))

    per_gpu: List[Dict[str, Any]] = []
    gpu_cycle: List[str] = []

    if auto:
        for di in device_indices:
            if di < 0 or di >= n_dev:
                continue
            try:
                if use_probe:
                    info = _probe_vram_slots(
                        di,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        feature_dim=feature_dim,
                        reserve_mb=reserve_mb,
                        process_overhead_mb=overhead_mb,
                    )
                else:
                    info = _heuristic_slots(di, est_gb_per_job=est_gb, reserve_gb=reserve_gb)
            except Exception as exc:
                info = _heuristic_slots(di, est_gb_per_job=est_gb, reserve_gb=reserve_gb)
                info["probe_error"] = str(exc)
            slots = max(1, int(info.get("slots", 1)))
            per_gpu.append({**info, "slots": slots})
            gpu_cycle.extend([str(di)] * slots)
        parallel = min(cap, max(1, len(gpu_cycle)))
        gpu_cycle = gpu_cycle[:parallel] if len(gpu_cycle) > parallel else gpu_cycle
        while len(gpu_cycle) < parallel and gpu_cycle:
            gpu_cycle.append(gpu_cycle[-1])
        if not gpu_cycle:
            gpu_cycle = ["0"]
            parallel = 1
        source = "auto_vram_probe" if use_probe else "auto_heuristic"
        report.update(
            {
                "source": source,
                "cuda_devices": n_dev,
                "per_gpu": per_gpu,
                "parallel": parallel,
                "gpu_cycle": list(gpu_cycle),
            }
        )
    else:
        parallel = max(1, min(cap, int(par.get("max_parallel_jobs", 1))))
        ids = [str(i) for i in device_indices] or ["0"]
        gpu_cycle = _expand_gpu_list(ids, parallel)
        report.update({"source": "config_manual", "parallel": parallel, "gpus": ids})

    dl = _dataloader_workers(par, cpus, parallel)
    report["dataloader_workers"] = dl
    return parallel, gpu_cycle, dl, report


def _expand_gpu_list(ids: List[str], parallel: int) -> List[str]:
    if not ids:
        ids = ["0"]
    out: List[str] = []
    for i in range(parallel):
        out.append(ids[i % len(ids)])
    return out


def _dataloader_workers(par: Dict[str, Any], cpus: int, parallel: int) -> int:
    raw = par.get("dataloader_workers", "auto")
    if raw == "auto" or raw is None:
        # Dejar CPUs para N procesos train + un poco de margen
        per_proc = max(2, min(8, cpus // max(1, parallel * 3)))
        return per_proc
    return max(0, int(raw) // max(1, parallel))


def format_parallel_report(report: Dict[str, Any]) -> str:
    lines = [
        f"  paralelo (jobs)     {report.get('parallel', '?')}",
        f"  origen              {report.get('source', '?')}",
    ]
    if report.get("per_gpu"):
        for g in report["per_gpu"]:
            extra = f" slots={g.get('slots')}" if g.get("slots") else ""
            lines.append(
                f"  GPU {g.get('device_index')} {g.get('device_name', '?')}"
                f"  VRAM≈{g.get('vram_total_mb', '?')} MiB{extra}"
            )
    if report.get("gpu_cycle"):
        lines.append(f"  asignación          {','.join(report['gpu_cycle'])}")
    lines.append(f"  DataLoader workers  {report.get('dataloader_workers', '?')}")
    return "\n".join(lines)