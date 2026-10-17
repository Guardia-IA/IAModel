"""Paralelismo de clips alineado con Triton (batch + GPU instances)."""
from __future__ import annotations

from triton_pose_extract.settings import ExtractSettings


def recommend_parallel_clips(settings: ExtractSettings) -> int:
    """
    Heurística offline (kanvis-triton-pose / C1 piloto):
    - `gpu_instance_count`: instancias TensorRT en Triton (p. ej. 2).
    - `max_batch_size`: 32 — límite de frames por inferencia agregada.
    - Cada clip envía lotes de `triton_batch_size` (p. ej. 8).

    No saturar la cola: ~min(gpu×2, max_batch // batch_size), al menos 1.
    """
    if settings.parallel_clips is not None:
        return max(1, int(settings.parallel_clips))
    gpu = max(1, int(settings.gpu_instance_count))
    batch = max(1, int(settings.triton_batch_size))
    max_bs = max(1, int(settings.max_batch_size))
    by_throughput = max(1, max_bs // batch)
    return max(1, min(gpu * 2, by_throughput))


def capacity_report(settings: ExtractSettings) -> dict:
    par = recommend_parallel_clips(settings)
    return {
        "parallel_clips": par,
        "triton_batch_size_per_clip": settings.triton_batch_size,
        "max_batch_size": settings.max_batch_size,
        "gpu_instance_count": settings.gpu_instance_count,
        "formula": "min(gpu_instance_count*2, max_batch_size//triton_batch_size)",
    }
