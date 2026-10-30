"""Motor ``c1_like`` pipelined: mismo grafo post-frame que el secuencial, con solape de etapas.

Paridad (contrato C1 / Gold N2) — lo que NO cambia respecto a ``extract_clip_c1_like``:

* Frames: ``cv2.VideoCapture.read()`` en orden (``iter_frames_opencv`` = ``_read_opencv``).
* Preprocess por frame: ``prepare_infer_image`` → ``letterbox`` → ``to_tensor`` (puro).
* Triton: chunks de ``triton_batch_size`` frames **consecutivos del mismo clip**, mismos
  límites de batch (0..7, 8..15, …, resto) que ``_infer_batched``. Nunca se mezclan clips en
  una request ni se reparte un chunk.
* Post-infer en orden estricto de frame: ``track_frame`` (decode_end2end → ByteTrack → filas).
* Export / filtros / meta: ``finalize_users_export``.

Lo que cambia (solo rendimiento / memoria):

* El decode no carga el clip entero en RAM: hilo productor → chunks → cola acotada.
* Preprocess y envío a Triton de los chunks k+1..k+N ocurren mientras el tracker consume k.
* Hasta ``infer_inflight`` requests gRPC concurrentes por clip (un canal por hilo).

Si el nº de frames decodificados no coincide con ffprobe, devuelve ``None`` y el llamador
repite el clip por el camino secuencial (que reconcilia backends igual que siempre).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from triton_pose_extract.extract_stack import PoseExtractStack, clone_triton_client
from triton_pose_extract.pipeline_config import C1LikeExtractConfig
from triton_pose_extract.video_decode import (
    ffprobe_nb_read_frames,
    ffprobe_nb_read_packets,
    iter_frames_opencv,
    probe_video,
)

logger = logging.getLogger(__name__)

DEFAULT_FPS = 12.0
_INFER_THREAD = threading.local()


def _ensure_pools(stack: PoseExtractStack, pipe_cfg: C1LikeExtractConfig) -> None:
    if stack.prep_pool is None:
        stack.prep_pool = ThreadPoolExecutor(
            max_workers=max(1, int(pipe_cfg.preprocess_workers)),
            thread_name_prefix="c1p_prep",
        )
    if stack.infer_pool is None:
        template = stack.client

        def _init() -> None:
            # Un canal gRPC por hilo; si no se puede clonar, comparte el del worker
            # (grpc es thread-safe; solo perdemos algo de paralelismo de red).
            try:
                _INFER_THREAD.client = clone_triton_client(template)
            except Exception as exc:  # noqa: BLE001
                logger.warning("infer pool: no se pudo clonar cliente Triton (%s); comparto", exc)
                _INFER_THREAD.client = template

        stack.infer_pool = ThreadPoolExecutor(
            max_workers=max(1, int(pipe_cfg.infer_inflight)),
            initializer=_init,
            thread_name_prefix="c1p_infer",
        )


def extract_clip_pipelined(
    stack: PoseExtractStack,
    video_path: Path,
    *,
    clip_root: Path,
    clip_name: str,
    cat: str | int | None,
    pipe_cfg: C1LikeExtractConfig,
    filters: Any | None,
) -> tuple[Any, dict[str, float]] | None:
    from pose_core.postprocess import split_triton_batch_output
    from pose_core.preprocess import letterbox, prepare_infer_image, to_tensor

    from triton_pose_extract.c1_like_extract import finalize_users_export, track_frame

    t0 = time.perf_counter()
    _ensure_pools(stack, pipe_cfg)
    cfg = stack.extract_cfg
    bs = max(1, int(cfg.triton_batch_size))
    inflight = max(1, int(pipe_cfg.infer_inflight))

    timing = probe_video(video_path)
    fps = timing.fps if timing.fps > 0 else DEFAULT_FPS
    t_probe0 = time.perf_counter()
    if pipe_cfg.ffprobe_fast_count:
        target = ffprobe_nb_read_packets(video_path)
        count_mode = "packets"
    else:
        target = ffprobe_nb_read_frames(video_path)
        count_mode = "frames"
    t_probe = time.perf_counter() - t_probe0

    def _prep_one(frame: np.ndarray):
        prep = prepare_infer_image(frame, stack.preprocess)
        boxed, lb_meta = letterbox(prep, stack.preprocess)
        return prep, lb_meta, to_tensor(boxed, stack.preprocess)

    def _infer_chunk(frames: list[np.ndarray]):
        # Preprocess del chunk (paralelo, orden preservado por map) + 1 request Triton.
        prepared = list(stack.prep_pool.map(_prep_one, frames))
        tensors = [p[2] for p in prepared]
        batch = np.stack(tensors, axis=0)
        client = getattr(_INFER_THREAD, "client", None) or stack.client
        t_i0 = time.perf_counter()
        out = client.infer(batch)
        t_i = time.perf_counter() - t_i0
        raws = split_triton_batch_output(out, len(frames))
        if len(raws) != len(frames):
            raise RuntimeError(
                f"Triton devolvió {len(raws)} tensors para batch_size={len(frames)}"
            )
        return [p[0] for p in prepared], [p[1] for p in prepared], raws, t_i

    # Cola de futures en orden de chunk; ``slots`` acota chunks en vuelo (memoria).
    chunk_q: queue.Queue[Future | None] = queue.Queue()
    slots = threading.BoundedSemaphore(inflight + 1)
    cancel = threading.Event()
    producer_err: list[BaseException] = []
    decode_wall: list[float] = [0.0]

    def _acquire_slot() -> bool:
        while not cancel.is_set():
            if slots.acquire(timeout=0.5):
                return True
        return False

    def _producer() -> None:
        t_d0 = time.perf_counter()
        try:
            frames: list[np.ndarray] = []
            for frame in iter_frames_opencv(video_path):
                frames.append(frame)
                if len(frames) == bs:
                    if not _acquire_slot():
                        return
                    chunk_q.put(stack.infer_pool.submit(_infer_chunk, frames))
                    frames = []
                if cancel.is_set():
                    return
            if frames and _acquire_slot():
                chunk_q.put(stack.infer_pool.submit(_infer_chunk, frames))
        except BaseException as exc:  # noqa: BLE001
            producer_err.append(exc)
        finally:
            decode_wall[0] = time.perf_counter() - t_d0
            chunk_q.put(None)

    stack.tracker.reset()
    users: dict[int, Any] = {}
    use_keep_all = cfg.frame_policy == "keep_all"
    decoded = 0
    t_wait = 0.0
    t_track = 0.0
    t_triton = 0.0
    n_chunks = 0

    prod = threading.Thread(target=_producer, name="c1p_decode", daemon=True)
    prod.start()
    try:
        while True:
            fut = chunk_q.get()
            if fut is None:
                break
            t_w0 = time.perf_counter()
            preps, metas, raws, t_i = fut.result()
            t_wait += time.perf_counter() - t_w0
            t_triton += t_i
            n_chunks += 1
            t_t0 = time.perf_counter()
            for prep, lb_meta, raw in zip(preps, metas, raws, strict=True):
                track_frame(stack, users, cfg, decoded, prep, lb_meta, raw, use_keep_all)
                decoded += 1
            t_track += time.perf_counter() - t_t0
            slots.release()
    except BaseException:
        cancel.set()
        raise
    finally:
        prod.join(timeout=30.0)

    if producer_err:
        raise producer_err[0]
    if decoded == 0:
        raise RuntimeError(f"0 frames: {video_path}")

    # Verificación T: packets (rápido) → si no cuadra, conteo autoritativo -count_frames.
    ffprobe_t = target
    if target is None or decoded != int(target):
        t_c0 = time.perf_counter()
        authoritative = ffprobe_nb_read_frames(video_path)
        t_probe += time.perf_counter() - t_c0
        if authoritative is not None:
            ffprobe_t = authoritative
            count_mode = "frames"
        if authoritative is None or decoded != int(authoritative):
            logger.warning(
                "%s: pipeline decode=%d != ffprobe packets=%s frames=%s → fallback secuencial",
                video_path.name,
                decoded,
                target,
                authoritative,
            )
            return None

    frame_count = timing.frame_count or decoded
    timings = {
        "probe_sec": round(t_probe, 3),
        "decode_wall_sec": round(decode_wall[0], 3),
        "infer_wait_sec": round(t_wait, 3),
        "triton_rtt_sec": round(t_triton, 3),
        "track_sec": round(t_track, 3),
    }
    return finalize_users_export(
        stack=stack,
        users=users,
        clip_root=clip_root,
        clip_name=clip_name,
        video_path=video_path,
        cat=cat,
        pipe_cfg=pipe_cfg,
        filters=filters,
        decoded_t=decoded,
        frame_count=frame_count,
        fps=fps,
        ffprobe_t=ffprobe_t,
        decode_backend_used="opencv",
        extra_meta={
            "iamodel_pipeline": True,
            "iamodel_infer_inflight": inflight,
            "iamodel_ffprobe_count_mode": count_mode,
            "iamodel_triton_chunks": n_chunks,
        },
        timings=timings,
        t0=t0,
    )
