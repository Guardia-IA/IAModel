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
    ok_coords = max_delta < epsilon if n_both > 0 else True
    ok_mask = vm_mismatch == 0
    return {
        "rows_compared": n,
        "valid_mask_mismatch": vm_mismatch,
        "rows_both_valid": n_both,
        "max_abs_delta_coords": max_delta,
        "epsilon": epsilon,
        "pass_mask": ok_mask,
        "pass_coords": ok_coords,
        "pass": ok_mask and ok_coords,
    }


def compare_merged_whole(
    c1_session: Path,
    offline_user: Path,
    epsilon: float,
) -> dict[str, Any]:
    merged_path = c1_session / "merged.npy"
    mvm_path = c1_session / "merged_valid_mask.npy"
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
    n = min(t_off, t_m)
    cmp = _compare_aligned_rows(off_pf, off_vm, merged, mvm, epsilon)
    result.update(cmp)
    if t_off != t_m:
        result["pass"] = False
        result["note"] = f"longitudes distintas (comparadas primeras {n} filas)"
    return result


def compare_chunk_timestamp_slice(
    c1_session: Path,
    offline_user: Path,
    chunk_id: str,
    epsilon: float,
) -> dict[str, Any]:
    chunk_dir = c1_session / chunk_id
    if not chunk_dir.is_dir():
        raise FileNotFoundError(f"chunk no encontrado: {chunk_dir}")
    meta_sess = json.loads((c1_session / "meta.json").read_text(encoding="utf-8"))
    playback_start = meta_sess.get("playback_start_at")
    if not playback_start:
        raise KeyError("meta.json sesión sin playback_start_at (§10.2)")
    t0_playback = _parse_iso_epoch(playback_start)

    video = c1_session / "clip_buffer.mp4"
    if not video.is_file():
        raise FileNotFoundError(f"falta clip_buffer.mp4 en {c1_session}")
    frame_count = _ffprobe_frame_count(video)
    fps_ref = float(meta_sess.get("pose_fps") or 0) or _ffprobe_fps(video)

    ts_path = chunk_dir / "pose_frame_timestamps.npy"
    chunk_pf = np.load(chunk_dir / "poses_full.npy")
    chunk_vm = np.load(chunk_dir / "valid_mask.npy")
    if not ts_path.is_file():
        raise FileNotFoundError(f"falta {ts_path}")
    ts = np.load(ts_path).astype(np.float64)

    off_pf = np.load(offline_user / "poses_full.npy")
    off_vm = np.load(offline_user / "valid_mask.npy")

    dt_tol = 0.5 / fps_ref
    indices: list[int] = []
    dt_violations = 0
    for t_i in ts:
        video_time_s = float(t_i - t0_playback)
        k = int(round(video_time_s * fps_ref))
        k = max(0, min(frame_count - 1, k))
        indices.append(k)
        t_impl = k / fps_ref
        if abs(video_time_s - t_impl) > dt_tol:
            dt_violations += 1

    idx = np.array(indices, dtype=np.int64)
    off_slice_pf = off_pf[idx]
    off_slice_vm = off_vm[idx]
    cmp = _compare_aligned_rows(off_slice_pf, off_slice_vm, chunk_pf, chunk_vm, epsilon)
    return {
        "test": "chunk_timestamp_slice",
        "chunk": chunk_id,
        "chunk_T": int(chunk_pf.shape[0]),
        "fps_ref": fps_ref,
        "frame_count": frame_count,
        "dt_alignment_violations": dt_violations,
        "dt_tol_seconds": dt_tol,
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
    args = ap.parse_args()

    offline_user = _pick_offline_user(args.offline_clip)
    report: dict[str, Any] = {
        "contract_version": "kanvis_pose_v1",
        "c1_session": str(args.c1_session.resolve()),
        "offline_clip": str(args.offline_clip.resolve()),
        "offline_user": offline_user.name,
        "tests": [],
    }

    if not args.skip_merged:
        report["tests"].append(
            compare_merged_whole(args.c1_session, offline_user, args.epsilon)
        )
    if not args.skip_chunk:
        report["tests"].append(
            compare_chunk_timestamp_slice(
                args.c1_session, offline_user, args.chunk, args.epsilon
            )
        )

    all_pass = all(t.get("pass") is True for t in report["tests"] if not t.get("skipped"))
    report["pass"] = all_pass

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
