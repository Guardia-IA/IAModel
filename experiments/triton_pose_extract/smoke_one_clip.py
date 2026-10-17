#!/usr/bin/env python3
"""Prueba rápida: 1 clip → out_triton_smoke/ con validación."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent


def _find_sample_video(explicit: Path | None) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if p.is_file():
            return p
        raise SystemExit(f"No existe vídeo: {p}")
    roots = [
        PKG / "samples",
        Path("/media/8TB/DatosEntrenamiento"),
    ]
    for root in roots:
        if not root.exists():
            continue
        for pat in ("**/clip.mp4", "**/*.mp4"):
            hits = sorted(root.glob(pat))
            if hits:
                return hits[0]
    raise SystemExit(
        "Pasa --video /ruta/clip.mp4 (no se encontró muestra en samples/ ni /media/8TB/...)"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default="")
    ap.add_argument("--out", type=str, default=str(PKG / "out_triton_smoke"))
    ap.add_argument("--triton-url", type=str, default="")
    ap.add_argument("--parallel", type=int, default=1)
    args = ap.parse_args()

    video = _find_sample_video(Path(args.video) if args.video else None)
    py = PKG / "extract_clips.py"
    cmd = [
        sys.executable,
        str(py),
        "check-triton",
    ]
    if args.triton_url:
        cmd.extend(["--triton-url", args.triton_url])
    print("== check-triton ==")
    subprocess.run(cmd, cwd=PKG, check=False)

    cmd = [
        sys.executable,
        str(py),
        "extract",
        "--video",
        str(video),
        "--out",
        args.out,
        "--validate",
        "--parallel",
        str(max(1, args.parallel)),
    ]
    if args.triton_url:
        cmd.extend(["--triton-url", args.triton_url])
    print("== extract ==")
    return subprocess.call(cmd, cwd=PKG)


if __name__ == "__main__":
    raise SystemExit(main())
