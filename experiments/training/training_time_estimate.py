"""
Estimación aproximada del tiempo de entrenamiento por experimento (model_config.EXPERIMENTS).

Usado por preflight_train_plan.py y reutilizable desde otros preflights.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

try:
    from .model_config import EXPERIMENTS  # type: ignore[attr-defined]
except ImportError:
    from model_config import EXPERIMENTS  # type: ignore[attr-defined]

# Segundos por batch (GPU potente ~5090); referencia empírica del preflight anterior.
BASE_TIME_PER_BATCH: Dict[str, float] = {
    "tcn": 0.004,
    "ms_tcn": 0.0058,
    "tcn_attn": 0.0060,
    "res_tcn": 0.005,
    "stgcn": 0.005,
    "gat_tcn": 0.0072,
    "lstm": 0.006,
    "gru": 0.0055,
    "gru_attn": 0.0062,
    "transformer": 0.008,
    "conformer_lite": 0.009,
    "pose_cnn2d": 0.006,
    "joint_attn": 0.008,
    "dilated_tcn": 0.005,
    "tcn_lstm": 0.007,
    "tcn_gru": 0.0065,
}
CPU_MULTIPLIER = 8.0
DEFAULT_BATCH_TIME = 0.006


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def fmt_duration(total_seconds: float) -> str:
    total_seconds = max(0.0, float(total_seconds))
    h = int(total_seconds // 3600)
    rem = total_seconds % 3600
    m = int(rem // 60)
    s = int(rem % 60)
    if h > 0:
        return f"~{h}h {m}m {s}s"
    if m > 0:
        return f"~{m}m {s}s"
    return f"~{s}s"


def estimate_single_experiment_seconds(
    cfg: Dict[str, Any],
    *,
    train_rows: int,
    val_rows: int,
) -> Optional[Dict[str, Any]]:
    if cfg.get("done", False):
        return None

    arch = str(cfg.get("arch", "tcn"))
    epochs = int(cfg.get("epochs", 20))
    batch_size = max(1, int(cfg.get("batch_size", 32)))
    seq_len = int(cfg.get("seq_len", 64))
    frame_factor = seq_len / 64.0

    train_batches = max(1, math.ceil(max(1, train_rows) / batch_size))
    val_batches = max(1, math.ceil(max(1, val_rows) / batch_size))
    batches_per_epoch = train_batches + val_batches

    t_gpu = BASE_TIME_PER_BATCH.get(arch, DEFAULT_BATCH_TIME) * frame_factor
    t_cpu = t_gpu * CPU_MULTIPLIER
    gpu_s = batches_per_epoch * t_gpu * epochs
    cpu_s = batches_per_epoch * t_cpu * epochs

    return {
        "arch": arch,
        "epochs": epochs,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "train_batches": train_batches,
        "val_batches": val_batches,
        "batches_per_epoch": batches_per_epoch,
        "gpu_seconds": float(gpu_s),
        "cpu_seconds": float(cpu_s),
    }


def estimate_all_experiments(
    *,
    train_rows: int,
    val_rows: int,
    experiments: Optional[List[Dict[str, Any]]] = None,
    experiment_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    exps = experiments if experiments is not None else EXPERIMENTS
    per_exp: List[Dict[str, Any]] = []
    total_gpu = 0.0
    total_cpu = 0.0
    pending = 0
    skipped = 0

    if experiment_ids is not None:
        sorted_pairs = sorted(
            (
                (int(eid), exps[int(eid) - 1])
                for eid in experiment_ids
                if 1 <= int(eid) <= len(exps)
            ),
            key=lambda p: (p[1].get("arch", ""), int(p[1].get("epochs", 0))),
        )
    else:
        sorted_pairs = sorted(
            enumerate(exps, start=1),
            key=lambda p: (p[1].get("arch", ""), int(p[1].get("epochs", 0))),
        )
    for exp_id, cfg in sorted_pairs:
        if cfg.get("done", False):
            skipped += 1
            continue
        est = estimate_single_experiment_seconds(cfg, train_rows=train_rows, val_rows=val_rows)
        if est is None:
            skipped += 1
            continue
        pending += 1
        total_gpu += est["gpu_seconds"]
        total_cpu += est["cpu_seconds"]
        per_exp.append({"exp_id": exp_id, "config": cfg, **est})

    on_gpu = cuda_available()
    primary_seconds = total_gpu if on_gpu else total_cpu
    return {
        "train_rows_per_epoch": int(train_rows),
        "val_rows_per_epoch": int(val_rows),
        "experiments_total": len(exps),
        "experiments_pending": pending,
        "experiments_skipped_done": skipped,
        "cuda_available": on_gpu,
        "total_gpu_seconds": float(total_gpu),
        "total_cpu_seconds": float(total_cpu),
        "primary_device": "gpu" if on_gpu else "cpu",
        "primary_total_seconds": float(primary_seconds),
        "per_experiment": per_exp,
    }


def format_estimate_report(
    summary: Dict[str, Any],
    *,
    bold: str = "",
    reset: str = "",
    cyan: str = "",
    yellow: str = "",
    header_fn=None,
    title: Optional[str] = None,
) -> None:
    if header_fn is not None:
        header_fn(
            title
            or "7) Estimación de tiempo (todos los experimentos en model_config.py)"
        )

    train_rows = summary["train_rows_per_epoch"]
    val_rows = summary["val_rows_per_epoch"]
    pending = summary["experiments_pending"]
    skipped = summary["experiments_skipped_done"]
    on_gpu = summary["cuda_available"]

    print(
        f"  Filas por época: train={cyan}{train_rows}{reset} | val={val_rows} "
        f"(test no cuenta en el bucle de entrenamiento)"
    )
    print(
        f"  Experimentos: {pending} pendientes (done=False) | {skipped} omitidos (done=True)"
    )
    print(
        f"  Dispositivo detectado: {cyan}{'GPU (CUDA)' if on_gpu else 'CPU (sin CUDA)'}{reset}"
    )
    print(
        f"  {yellow}Nota: estimación heurística; depende de GPU, workers, I/O y carga real.{reset}"
    )

    print(
        f"\n{bold}{'ID':>3} | {'Arch':>14} | {'Epochs':>6} | {'Batch':>5} | "
        f"{'SeqLen':>6} | {'GPU(min)':>9} | {'CPU(min)':>9}{reset}"
    )
    print("-" * 90)

    for item in summary["per_experiment"]:
        exp_id = item["exp_id"]
        print(
            f"{exp_id:3d} | {item['arch']:>14} | {item['epochs']:6d} | "
            f"{item['batch_size']:5d} | {item['seq_len']:6d} | "
            f"{item['gpu_seconds'] / 60.0:9.2f} | {item['cpu_seconds'] / 60.0:9.2f}"
        )

    print(f"\n{bold}Tiempo TOTAL estimado (solo experimentos pendientes):{reset}")
    print(f"  - GPU: {cyan}{fmt_duration(summary['total_gpu_seconds'])}{reset} "
          f"({summary['total_gpu_seconds'] / 3600.0:.1f} h)")
    print(f"  - CPU: {cyan}{fmt_duration(summary['total_cpu_seconds'])}{reset} "
          f"({summary['total_cpu_seconds'] / 3600.0:.1f} h)")
    if on_gpu:
        print(
            f"  → En tu máquina (CUDA): {cyan}{fmt_duration(summary['total_gpu_seconds'])}{reset}"
        )
    else:
        print(
            f"  → En tu máquina (sin CUDA): {cyan}{fmt_duration(summary['total_cpu_seconds'])}{reset}"
        )
