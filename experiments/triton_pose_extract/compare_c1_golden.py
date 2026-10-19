#!/usr/bin/env python3
"""
Gold tests C1 MinIO export vs offline ClipExtractor (kanvis_pose_v1 Nivel 1).

Spec: IAModel/docs/MENSAJE_IAMODEL_PARIDAD_POSES_FULL_C1.md §8, §10.2–§10.3.

Usage:
  # Tras re-extract del clip_buffer:
  python compare_c1_golden.py \\
    --c1-session /path/to/session_pose_export_UUID \\
    --offline-clip /path/to/data_result/.../clip_name \\
    [--chunk chunk_001] [--epsilon 1e-4] [--json report.json]

  --offline-clip: carpeta con user_*/poses_full.npy (salida triton_pose_extract).
  --c1-session: raíz export sesión (clip_buffer.mp4, merged*.npy, meta.json, chunk_*).
  Gold principal: merged.npy ↔ offline (mismo clip_buffer). Chunk usa pose_frame_timestamps
  + merged_pose_timestamps.npy (sin frontend_playback).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

EPS_DEFAULT = 1e-4


def _parse_iso_epoch(s: str) -> float:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _resolve_playback_t0(c1_session: Path, meta_sess: dict[str, Any]) -> tuple[float, str]:
    """
    Epoch UTC del instante t=0 del clip_buffer (§10.2).
    Algunos exports MinIO no incluyen playback_start_at en meta.json raíz.
    """
    for key in ("playback_start_at", "clip_playback_start_at", "video_playback_start_at"):
        raw = meta_sess.get(key)
        if raw:
            return _parse_iso_epoch(str(raw)), f"meta.json:{key}"

    sync = meta_sess.get("sync")
    if isinstance(sync, dict):
        for key in ("playback_start_at", "video_start_at"):
            raw = sync.get(key)
            if raw:
                return _parse_iso_epoch(str(raw)), f"meta.json:sync.{key}"

    fb_path = c1_session / "frontend_playback.json"
    if fb_path.is_file():
        fb = json.loads(fb_path.read_text(encoding="utf-8"))
        sync_fb = fb.get("sync") if isinstance(fb.get("sync"), dict) else fb
        if isinstance(sync_fb, dict):
            for key in ("playback_start_at", "video_start_at"):
                raw = sync_fb.get(key)
                if raw:
                    return _parse_iso_epoch(str(raw)), f"frontend_playback.json:sync.{key}"
            pts = sync_fb.get("pose_timeline_start_at")
            if pts:
                offset = float(
                    sync_fb.get("pose_offset_seconds")
                    or sync_fb.get("overlay_start_video_time_s")
                    or fb.get("pose_offset_seconds")
                    or 0.0
                )
                t0 = _parse_iso_epoch(str(pts)) - offset
                return t0, "frontend_playback.json:pose_timeline_start_at - offset"

    chunk_meta_path = c1_session / "chunk_001" / "meta.json"
    if chunk_meta_path.is_file():
        cm = json.loads(chunk_meta_path.read_text(encoding="utf-8"))
        for key in ("playback_start_at", "session_playback_start_at"):
            raw = cm.get(key)
            if raw:
                return _parse_iso_epoch(str(raw)), f"chunk_001/meta.json:{key}"

    pts = meta_sess.get("pose_timeline_start_at")
    if pts:
        offset = float(meta_sess.get("pose_offset_seconds") or 0.0)
        t0 = _parse_iso_epoch(str(pts)) - offset
        return t0, "meta.json:pose_timeline_start_at - pose_offset_seconds"

    raise KeyError(
        "No se encontró playback_start_at (meta.json, frontend_playback.json sync, "
        "o pose_timeline_start_at + pose_offset_seconds). §10.2"
    )


def _has_playback_sync_start(c1_session: Path, meta_sess: dict[str, Any]) -> bool:
    try:
        _resolve_playback_t0(c1_session, meta_sess)
        return True
    except KeyError:
        return False


def _ffprobe_fps(video: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,nb_frames",
        "-of",
        "json",
        str(video),
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe sin stream de vídeo: {video}")
    st = streams[0]
    fr = st.get("avg_frame_rate") or "0/0"
    num, den = fr.split("/")
    num, den = int(num), int(den)
    if den <= 0:
        raise RuntimeError(f"avg_frame_rate inválido: {fr}")
    return num / den


def _ffprobe_frame_count(video: Path) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video),
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        if out.isdigit():
            return int(out)
    except subprocess.CalledProcessError:
        pass
    fps = _ffprobe_fps(video)
    cmd2 = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video),
    ]
    dur = float(subprocess.check_output(cmd2, text=True).strip())
    return max(1, int(round(dur * fps)))


def _pick_offline_user(offline_root: Path) -> Path:
    users = sorted(offline_root.glob("user_*"))
    if not users:
        raise FileNotFoundError(f"sin user_* en {offline_root}")
    if len(users) == 1:
        return users[0]
    best = max(users, key=lambda p: np.load(p / "valid_mask.npy").sum())
    return best


def _compare_aligned_rows(
    a_full: np.ndarray,
    a_vm: np.ndarray,
    b_full: np.ndarray,
    b_vm: np.ndarray,
    epsilon: float,
    *,
    require_rows_both_valid: bool = False,
) -> dict[str, Any]:
    n = min(len(a_vm), len(b_vm))
    a_vm, b_vm = a_vm[:n], b_vm[:n]
    a_full, b_full = a_full[:n], b_full[:n]
    vm_mismatch = int(np.sum(a_vm != b_vm))
    both = a_vm & b_vm
    n_both = int(both.sum())
    max_delta = 0.0
    if n_both > 0:
        diff = np.abs(a_full[both] - b_full[both])
        max_delta = float(np.nanmax(diff))
    if n_both > 0:
        ok_coords = max_delta < epsilon
    elif require_rows_both_valid:
        ok_coords = False
    else:
        ok_coords = True
    ok_mask = vm_mismatch == 0
    out: dict[str, Any] = {
        "rows_compared": n,
        "valid_mask_mismatch": vm_mismatch,
        "rows_both_valid": n_both,
        "max_abs_delta_coords": max_delta,
        "epsilon": epsilon,
        "pass_mask": ok_mask,
        "pass_coords": ok_coords,
        "pass": ok_mask and ok_coords,
    }
    if require_rows_both_valid and n_both == 0:
        out["note"] = (
            "Sin filas both-valid: chunk vacío/NaN o alineación temporal incorrecta — no es PASS N2."
        )
    return out


def _offline_indices_for_merged_timeline(
    mts: np.ndarray,
    off_len: int,
    fps: float,
    merged: np.ndarray,
    mvm: np.ndarray,
    off_pf: np.ndarray,
    off_vm: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """
    merged parcial (RTSP) vs offline denso: busca j0 en offline vía merged_pose_timestamps.
    Nunca lanza: si no hay ancla útil, devuelve align_failed en meta.
    """
    t_m = int(len(mts))
    if t_m == 0 or off_len == 0:
        return None, {
            "align_mode": "merged_pose_timestamps",
            "align_failed": True,
            "note": "merged u offline vacío (T=0)",
            "rows_compared": 0,
            "valid_mask_mismatch": 0,
            "rows_both_valid": 0,
            "max_abs_delta_coords": 0.0,
            "epsilon": epsilon,
            "pass_mask": False,
            "pass_coords": False,
            "pass": False,
        }

    rel_s = (mts.astype(np.float64) - float(mts[0])) / max(fps, 1e-6)
    span_frames = int(np.ceil(float(rel_s[-1]))) + 1 if t_m else 0
    j0_max = max(0, off_len - span_frames - 1)
    best_j0 = 0
    best_cmp: dict[str, Any] | None = None
    best_score = float("inf")
    best_idx: np.ndarray | None = None

    for j0 in range(0, j0_max + 1):
        idx = np.array(
            [max(0, min(off_len - 1, j0 + int(round(rel_s[i])))) for i in range(t_m)],
            dtype=np.int64,
        )
        cmp = _compare_aligned_rows(off_pf[idx], off_vm[idx], merged, mvm, epsilon)
        if cmp["rows_both_valid"] == 0:
            score = float("inf")
        else:
            score = float(cmp["max_abs_delta_coords"])
        if score < best_score:
            best_score = score
            best_j0 = j0
            best_cmp = cmp
            best_idx = idx

    meta: dict[str, Any] = {
        "align_mode": "merged_pose_timestamps",
        "offline_j0": int(best_j0),
        "merged_span_seconds": round(float(mts[-1] - mts[0]), 4) if t_m > 1 else 0.0,
    }
    if best_cmp is None or best_idx is None or best_score == float("inf"):
        meta.update(
            {
                "align_failed": True,
                "note": (
                    "No se encontró ancla merged_ts → offline con filas both-valid "
                    f"(merged_T={t_m}, offline_T={off_len}). Posible merged RTSP/regen "
                    "sobre export denso N2 — ver C1 frontend_playback_bundle_regenerated."
                ),
                "rows_compared": 0,
                "valid_mask_mismatch": 0,
                "rows_both_valid": 0,
                "max_abs_delta_coords": 0.0,
                "epsilon": epsilon,
                "pass_mask": False,
                "pass_coords": False,
                "pass": False,
            }
        )
        return best_idx, meta
    return best_idx, {**meta, **best_cmp}


def compare_merged_whole(
    c1_session: Path,
    offline_user: Path,
    epsilon: float,
    *,
    merged_align: str = "auto",
) -> dict[str, Any]:
    merged_path = c1_session / "merged.npy"
    mvm_path = c1_session / "merged_valid_mask.npy"
    mts_path = c1_session / "merged_pose_timestamps.npy"
    if not merged_path.is_file():
        return {"skipped": True, "reason": "falta merged.npy"}
    off_pf = np.load(offline_user / "poses_full.npy")
    off_vm = np.load(offline_user / "valid_mask.npy")
    merged = np.load(merged_path)
    if not mvm_path.is_file():
        return {"skipped": True, "reason": "falta merged_valid_mask.npy"}
    mvm = np.load(mvm_path)
    t_off, t_m = off_pf.shape[0], merged.shape[0]
    result: dict[str, Any] = {
        "test": "merged_whole_clip",
        "offline_T": t_off,
        "merged_T": t_m,
        "T_equal": t_off == t_m,
    }

    meta_sess = {}
    meta_path = c1_session / "meta.json"
    if meta_path.is_file():
        meta_sess = json.loads(meta_path.read_text(encoding="utf-8"))
    video = c1_session / "clip_buffer.mp4"
    fps_ref = float(meta_sess.get("pose_fps") or 0)
    if fps_ref <= 0 and video.is_file():
        fps_ref = _ffprobe_fps(video)
    if fps_ref <= 0:
        fps_ref = 12.0
    if video.is_file():
        result["ffprobe_nb_read_frames"] = _ffprobe_frame_count(video)
        result["ffprobe_T_equal_offline"] = result["ffprobe_nb_read_frames"] == t_off
        result["ffprobe_T_equal_merged"] = result["ffprobe_nb_read_frames"] == t_m
        if not result["ffprobe_T_equal_offline"] and result.get("ffprobe_T_equal_merged"):
            result.setdefault(
                "note",
                "offline_T != ffprobe: re-ejecutar triton_pose_extract (--validate) "
                "tras actualizar decode (reconcile ffprobe nb_read_frames).",
            )

    mode = merged_align
    if mode == "auto":
        mode = "dense_index" if t_off == t_m else "merged_ts"

    # Nivel 2: T igual → fila i ↔ fila i (re-infer denso vs offline)
    if mode == "dense_index" and t_off == t_m:
        cmp = _compare_aligned_rows(off_pf, off_vm, merged, mvm, epsilon)
        result.update(cmp)
        result["align_mode"] = "dense_index"
        result["pass_strict_T_equal"] = True
        off_meta_path = offline_user.parent / "meta.json"
        if off_meta_path.is_file():
            off_meta = json.loads(off_meta_path.read_text(encoding="utf-8"))
            result["offline_decode_backend"] = off_meta.get("decode_backend_used") or off_meta.get(
                "iamodel_decode_backend"
            )
        result["note"] = "N2: merged_T == offline_T; comparación índice i (§8 gold denso)."
        if not cmp.get("pass_coords"):
            result["note"] += (
                " Δ coords alto con máscaras OK → offline debe inferir con OpenCV "
                "(decode_c1_pixel_parity); re-extract y git pull IAModel."
            )
        return result

    if mode == "merged_ts" and mts_path.is_file() and t_off != t_m:
        mts = np.load(mts_path).astype(np.float64)
        if len(mts) != t_m:
            result["skipped"] = True
            result["reason"] = f"merged_pose_timestamps len {len(mts)} != merged_T {t_m}"
            return result
        _idx, cmp_meta = _offline_indices_for_merged_timeline(
            mts, t_off, fps_ref, merged, mvm, off_pf, off_vm, epsilon
        )
        result.update(cmp_meta)
        aligned_ok = bool(
            cmp_meta.get("pass_mask")
            and cmp_meta.get("pass_coords")
            and not cmp_meta.get("align_failed")
        )
        result["pass_aligned"] = aligned_ok
        result["pass"] = aligned_ok and (t_off == t_m)
        result["pass_strict_T_equal"] = t_off == t_m
        if t_off != t_m:
            result["note"] = (
                "merged no cubre todo el clip_buffer (RTSP/ventana o regen roto). "
                "Gold §8 N2 exige T_merged == T_offline (re-infer clip_buffer denso). "
                "Si merged_T < offline_T, revisar frontend_playback_bundle_regenerated."
            )
        return result

    if t_off == t_m:
        cmp = _compare_aligned_rows(off_pf, off_vm, merged, mvm, epsilon)
        result.update(cmp)
        result["align_mode"] = "dense_index"
        result["pass_strict_T_equal"] = True
        return result

    if t_off != t_m:
        n = min(t_off, t_m)
        cmp = _compare_aligned_rows(
            off_pf[:n], off_vm[:n], merged[:n], mvm[:n], epsilon
        )
        result.update(cmp)
        result["pass"] = False
        result["pass_strict_T_equal"] = False
        result["align_mode"] = "head_rows_unaligned"
        result["note"] = (
            f"T distinto ({t_m} vs {t_off}). Comparación fila 0..{n - 1} NO válida para N2; "
            "usar merged denso T=offline o merged_ts si hay timestamps."
        )
        return result

    cmp = _compare_aligned_rows(off_pf, off_vm, merged, mvm, epsilon)
    result.update(cmp)
    result["align_mode"] = "dense_index"
    return result


def _chunk_offline_indices(
    c1_session: Path,
    chunk_dir: Path,
    meta_sess: dict[str, Any],
    *,
    off_len: int,
    align: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Índices en offline poses_full por fila del chunk.
    Gold chunk (§10.2): align=playback — pose_frame_timestamps → tiempo en clip_buffer → k_i.
    merged_ts: empareja ts del chunk con merged_pose_timestamps (no sustituye §10.2).
    """
    ts_path = chunk_dir / "pose_frame_timestamps.npy"
    if not ts_path.is_file():
        raise FileNotFoundError(f"falta {ts_path}")
    ts = np.load(ts_path).astype(np.float64)

    video = c1_session / "clip_buffer.mp4"
    fps_ref = float(meta_sess.get("pose_fps") or 0)
    if fps_ref <= 0 and video.is_file():
        fps_ref = _ffprobe_fps(video)
    if fps_ref <= 0:
        fps_ref = 12.0
    dt_tol = 0.5 / fps_ref

    meta_align: dict[str, Any] = {"align_mode": align, "fps_ref": fps_ref, "dt_tol_seconds": dt_tol}
    indices: list[int] = []
    dt_violations = 0

    mode = align
    mts_path = c1_session / "merged_pose_timestamps.npy"
    if mode == "merged_ts" and not mts_path.is_file():
        mode = "relative"
        meta_align["align_fallback"] = "merged_pose_timestamps.npy ausente → relative"

    if mode == "merged_ts":
        mts = np.load(mts_path).astype(np.float64)
        n_ref = min(len(mts), off_len)
        meta_align["align_mode"] = "merged_ts"
        meta_align["merged_ts_len"] = int(len(mts))
        for t_i in ts:
            j = int(np.argmin(np.abs(mts[:n_ref] - t_i)))
            if abs(mts[j] - t_i) > dt_tol:
                dt_violations += 1
            indices.append(max(0, min(off_len - 1, j)))
    elif mode == "relative":
        meta_align["align_mode"] = "relative"
        t0 = float(ts[0]) if len(ts) else 0.0
        for t_i in ts:
            k = int(round((float(t_i) - t0) * fps_ref))
            indices.append(max(0, min(off_len - 1, k)))
    elif mode == "playback":
        t0_playback, t0_source = _resolve_playback_t0(c1_session, meta_sess)
        meta_align["align_mode"] = "playback"
        meta_align["playback_t0_source"] = t0_source
        meta_align["playback_t0_epoch"] = t0_playback
        if video.is_file():
            meta_align["ffprobe_frame_count"] = _ffprobe_frame_count(video)
        for t_i in ts:
            video_time_s = float(t_i - t0_playback)
            k = int(round(video_time_s * fps_ref))
            indices.append(max(0, min(off_len - 1, k)))
            t_impl = k / fps_ref
            if abs(video_time_s - t_impl) > dt_tol:
                dt_violations += 1
    elif mode == "merged_dense":
        if not mts_path.is_file():
            raise FileNotFoundError(
                "align merged_dense requiere merged_pose_timestamps.npy (N2 denso)"
            )
        mts = np.load(mts_path).astype(np.float64)
        meta_align["align_mode"] = "merged_dense"
        meta_align["merged_ts_len"] = int(len(mts))
        meta_align["note"] = (
            "Índice offline = argmin merged_pose_timestamps (sin playback_start_at). "
            "Válido si offline_T == merged_T == ffprobe."
        )
        for t_i in ts:
            j = int(np.argmin(np.abs(mts - t_i)))
            if abs(mts[j] - t_i) > dt_tol:
                dt_violations += 1
            indices.append(max(0, min(off_len - 1, j)))
    else:
        raise ValueError(f"align desconocido: {align}")

    meta_align["dt_alignment_violations"] = dt_violations
    return np.array(indices, dtype=np.int64), meta_align


def compare_chunk_timestamp_slice(
    c1_session: Path,
    offline_user: Path,
    chunk_id: str,
    epsilon: float,
    *,
    align: str = "playback",
) -> dict[str, Any]:
    chunk_dir = c1_session / chunk_id
    if not chunk_dir.is_dir():
        raise FileNotFoundError(f"chunk no encontrado: {chunk_dir}")
    meta_sess = json.loads((c1_session / "meta.json").read_text(encoding="utf-8"))

    chunk_pf = np.load(chunk_dir / "poses_full.npy")
    chunk_vm = np.load(chunk_dir / "valid_mask.npy")
    off_pf = np.load(offline_user / "poses_full.npy")
    off_vm = np.load(offline_user / "valid_mask.npy")
    meta_align_pre: dict[str, Any] = {
        "has_sync_start": _has_playback_sync_start(c1_session, meta_sess),
    }

    idx, meta_align = _chunk_offline_indices(
        c1_session,
        chunk_dir,
        meta_sess,
        off_len=int(off_pf.shape[0]),
        align=align,
    )
    off_slice_pf = off_pf[idx]
    off_slice_vm = off_vm[idx]
    cmp = _compare_aligned_rows(
        off_slice_pf,
        off_slice_vm,
        chunk_pf,
        chunk_vm,
        epsilon,
        require_rows_both_valid=True,
    )
    return {
        "test": "chunk_timestamp_slice",
        "chunk": chunk_id,
        "chunk_T": int(chunk_pf.shape[0]),
        "offline_T": int(off_pf.shape[0]),
        **meta_align_pre,
        **meta_align,
        **cmp,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Gold C1 vs offline (kanvis_pose_v1)")
    ap.add_argument("--c1-session", type=Path, required=True)
    ap.add_argument("--offline-clip", type=Path, required=True)
    ap.add_argument("--chunk", default="chunk_001", help="chunk para test §10.2")
    ap.add_argument("--epsilon", type=float, default=EPS_DEFAULT)
    ap.add_argument("--json", type=Path, help="escribir informe JSON")
    ap.add_argument("--skip-merged", action="store_true")
    ap.add_argument("--skip-chunk", action="store_true")
    ap.add_argument(
        "--align",
        choices=("auto", "playback", "merged_ts", "relative", "dense_index", "merged_dense"),
        default="playback",
        help=(
            "Chunk↔offline: playback=§10.2 (default). auto=playback si sync, si no merged_dense. "
            "merged_dense=N2 sin playback_start_at. dense_index si T chunk==offline"
        ),
    )
    ap.add_argument(
        "--merged-align",
        choices=("auto", "dense_index", "merged_ts", "head"),
        default="auto",
        help="merged↔offline: auto (=dense_index si T igual, si no merged_ts)",
    )
    args = ap.parse_args()

    try:
        offline_user = _pick_offline_user(args.offline_clip)
    except FileNotFoundError as exc:
        report = {
            "contract_version": "kanvis_pose_v1",
            "c1_session": str(args.c1_session.resolve()),
            "offline_clip": str(args.offline_clip.resolve()),
            "pass": False,
            "error": str(exc),
            "tests": [],
        }
        text = json.dumps(report, indent=2, ensure_ascii=False)
        print(text)
        if args.json:
            args.json.write_text(text + "\n", encoding="utf-8")
        return 1

    report: dict[str, Any] = {
        "contract_version": "kanvis_pose_v1",
        "c1_session": str(args.c1_session.resolve()),
        "offline_clip": str(args.offline_clip.resolve()),
        "offline_user": offline_user.name,
        "tests": [],
    }

    if not args.skip_merged:
        try:
            report["tests"].append(
                compare_merged_whole(
                    args.c1_session,
                    offline_user,
                    args.epsilon,
                    merged_align=args.merged_align,
                )
            )
        except Exception as exc:
            report["tests"].append(
                {
                    "test": "merged_whole_clip",
                    "pass": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if not args.skip_chunk:
        try:
            chunk_align = args.align
            meta_sess = json.loads((args.c1_session / "meta.json").read_text(encoding="utf-8"))
            if chunk_align == "auto":
                chunk_align = (
                    "playback"
                    if _has_playback_sync_start(args.c1_session, meta_sess)
                    else "merged_dense"
                )
                report.setdefault("warnings", []).append(
                    f"align auto → {chunk_align} "
                    f"(has_sync_start={_has_playback_sync_start(args.c1_session, meta_sess)})"
                )
            if chunk_align == "dense_index":
                chunk_pf = np.load(
                    (args.c1_session / args.chunk / "poses_full.npy")
                )
                off_t = np.load(offline_user / "poses_full.npy").shape[0]
                if chunk_pf.shape[0] != off_t:
                    chunk_align = "playback"
                    report.setdefault("warnings", []).append(
                        "dense_index chunk omitido: T chunk != T offline; usando playback"
                    )
            report["tests"].append(
                compare_chunk_timestamp_slice(
                    args.c1_session,
                    offline_user,
                    args.chunk,
                    args.epsilon,
                    align=chunk_align,
                )
            )
        except Exception as exc:
            report["tests"].append(
                {
                    "test": "chunk_timestamp_slice",
                    "chunk": args.chunk,
                    "pass": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    active = [t for t in report["tests"] if not t.get("skipped")]
    merged_ok = next(
        (t.get("pass") is True for t in active if t.get("test") == "merged_whole_clip"),
        False,
    )
    chunk_ok = next(
        (t.get("pass") is True for t in active if t.get("test") == "chunk_timestamp_slice"),
        True,
    )
    if args.skip_chunk:
        report["pass"] = merged_ok
    elif args.skip_merged:
        report["pass"] = chunk_ok
    else:
        report["pass"] = merged_ok and chunk_ok
    report["pass_merged"] = merged_ok
    report["pass_chunk"] = chunk_ok if not args.skip_chunk else None

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")

    return 0 if report.get("pass") else 1


if __name__ == "__main__":
    sys.exit(main())
