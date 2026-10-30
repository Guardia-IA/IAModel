#!/usr/bin/env python3
"""A/B de paridad: motor c1_like secuencial (TRITON_EXTRACT_PIPELINE=0) vs pipelined (=1).

Extrae el/los mismos clips dos veces con ``extract_clips.py extract`` (misma config, mismo
Triton) y compara bit a bit ``poses_full.npy``, ``valid_mask.npy``, ``keypoint_mask.npy`` y
``poses.npy`` de cada ``user_*``, además de ``frame_count`` y ``decode_backend_used`` en meta.

Uso (wudebian, Triton up, venv activo, KANVIS_TRITON_POSE_ROOT exportado):

  cd ~/IAModel/experiments/triton_pose_extract
  ./run_extract.sh parity-ab --video /ruta/clip.mp4 [--video otro.mp4 ...]
  ./run_extract.sh parity-ab --video-dir /ruta/carpeta_mp4 --limit 10

Criterio PASS: todos los arrays idénticos (NaN en las mismas posiciones) y mismo T.
Si quieres también medir tiempo: cada pasada imprime su wall.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]


def _run_extract(video: Path, out: Path, *, pipeline: bool, extra: list[str]) -> float:
    env = dict(os.environ)
    env["TRITON_EXTRACT_PIPELINE"] = "1" if pipeline else "0"
    env["PYTHONPATH"] = str(PKG) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        str(PKG / "extract_clips.py"),
        "extract",
        "--video",
        str(video),
        "--out",
        str(out),
        "--cat",
        "ab",
        "--parallel",
        "1",
        *extra,
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(PKG), env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-2000:] + "\n" + proc.stderr[-3000:] + "\n")
        raise RuntimeError(f"extract exit {proc.returncode} (pipeline={pipeline})")
    return wall


def _clip_dir(out: Path, video: Path) -> Path:
    root = out / "ab"
    cands = [p for p in root.iterdir() if p.is_dir()] if root.is_dir() else []
    if len(cands) == 1:
        return cands[0]
    for p in cands:
        if p.name == video.stem:
            return p
    raise FileNotFoundError(f"clip dir no encontrado bajo {root}: {cands}")


def _arrays_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if np.issubdtype(a.dtype, np.floating):
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _compare_clip(seq_dir: Path, pipe_dir: Path) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok = True
    meta_s = json.loads((seq_dir / "meta.json").read_text(encoding="utf-8"))
    meta_p = json.loads((pipe_dir / "meta.json").read_text(encoding="utf-8"))
    for key in ("frame_count", "decode_backend_used", "poses_full_T"):
        if meta_s.get(key) != meta_p.get(key):
            ok = False
            notes.append(f"meta.{key}: seq={meta_s.get(key)} pipe={meta_p.get(key)}")
    notes.append(
        f"T={meta_s.get('frame_count')} ffprobe seq={meta_s.get('ffprobe_nb_read_frames')} "
        f"pipe={meta_p.get('ffprobe_nb_read_frames')} count_mode={meta_p.get('iamodel_ffprobe_count_mode')}"
    )
    users_s = sorted(p.name for p in seq_dir.glob("user_*"))
    users_p = sorted(p.name for p in pipe_dir.glob("user_*"))
    if users_s != users_p:
        ok = False
        notes.append(f"user_* distintos: seq={users_s} pipe={users_p}")
    for u in users_s:
        if u not in users_p:
            continue
        for fname in ("poses_full.npy", "valid_mask.npy", "keypoint_mask.npy", "poses.npy"):
            fs, fp = seq_dir / u / fname, pipe_dir / u / fname
            if not fs.is_file() or not fp.is_file():
                if fs.is_file() != fp.is_file():
                    ok = False
                    notes.append(f"{u}/{fname}: existe seq={fs.is_file()} pipe={fp.is_file()}")
                continue
            a, b = np.load(fs), np.load(fp)
            if _arrays_equal(a, b):
                notes.append(f"{u}/{fname}: idéntico {a.shape}")
            else:
                ok = False
                if a.shape == b.shape and np.issubdtype(a.dtype, np.floating):
                    d = np.nanmax(np.abs(a - b)) if a.size else 0.0
                    nan_mm = int(np.sum(np.isnan(a) != np.isnan(b)))
                    notes.append(f"{u}/{fname}: DIFF max|Δ|={d:.3g} nan_mismatch={nan_mm}")
                else:
                    notes.append(f"{u}/{fname}: DIFF shape/dtype seq={a.shape}{a.dtype} pipe={b.shape}{b.dtype}")
    return ok, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", action="append", default=[], help="MP4 (repetible)")
    ap.add_argument("--video-dir", type=str, default="", help="Carpeta con *.mp4")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep", action="store_true", help="No borrar salidas temporales")
    ap.add_argument("--work", type=str, default="", help="Carpeta de trabajo (default: mkdtemp)")
    ap.add_argument("extra", nargs="*", help="Args extra para extract (tras --)")
    args = ap.parse_args()

    videos = [Path(v).expanduser().resolve() for v in args.video]
    if args.video_dir:
        videos += sorted(Path(args.video_dir).expanduser().resolve().glob("*.mp4"))
    if args.limit > 0:
        videos = videos[: args.limit]
    if not videos:
        print("ERROR: indica --video o --video-dir", file=sys.stderr)
        return 2

    work = Path(args.work).expanduser().resolve() if args.work else Path(tempfile.mkdtemp(prefix="parity_ab_"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"parity-ab: {len(videos)} clip(s) · work={work}", flush=True)

    all_ok = True
    rows = []
    for i, video in enumerate(videos, 1):
        out_seq = work / f"{i:03d}_seq"
        out_pipe = work / f"{i:03d}_pipe"
        print(f"\n[{i}/{len(videos)}] {video.name}", flush=True)
        try:
            w_seq = _run_extract(video, out_seq, pipeline=False, extra=args.extra)
            w_pipe = _run_extract(video, out_pipe, pipeline=True, extra=args.extra)
            ok, notes = _compare_clip(_clip_dir(out_seq, video), _clip_dir(out_pipe, video))
        except Exception as exc:  # noqa: BLE001
            ok, notes, w_seq, w_pipe = False, [f"ERROR: {exc}"], 0.0, 0.0
        all_ok &= ok
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] wall seq={w_seq:.1f}s pipe={w_pipe:.1f}s", flush=True)
        for n in notes:
            print(f"    · {n}", flush=True)
        rows.append({"video": str(video), "pass": ok, "wall_seq": w_seq, "wall_pipe": w_pipe, "notes": notes})

    (work / "parity_ab_report.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nResumen: {'PASS' if all_ok else 'FAIL'} ({sum(r['pass'] for r in rows)}/{len(rows)}) · informe {work / 'parity_ab_report.json'}")
    if not args.keep and all_ok and not args.work:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
