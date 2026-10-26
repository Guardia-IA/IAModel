#!/usr/bin/env python3
"""Comprobaciones C1 cierre sesión (--session-close): T, valid_mask, meta."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_root", type=Path, help="Ej. …/gold/clip o …/gold/clip_buffer")
    args = ap.parse_args()
    root = args.clip_root.expanduser().resolve()
    meta_path = root / "meta.json"
    if not meta_path.is_file():
        print(f"ERROR: no {meta_path}", file=sys.stderr)
        return 2

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fc = int(meta.get("frame_count") or meta.get("video_frame_count") or 0)
    users = list(root.glob("user_*"))
    user_dirs = [p for p in users if p.is_dir()]

    print("=== meta.json (resumen) ===")
    print(json.dumps(
        {
            k: meta.get(k)
            for k in (
                "clip_name",
                "frame_count",
                "frame_policy",
                "iamodel_session_close_export",
                "iamodel_export_track_id",
                "iamodel_primary_user_track_id",
                "users",
            )
            if k in meta
        },
        indent=2,
        ensure_ascii=False,
    ))

    print(f"\nuser_* dirs: {len(user_dirs)} → {[p.name for p in user_dirs]}")
    if len(user_dirs) != 1:
        print(f"FAIL: se esperaba exactamente 1 user_* (session-close + primary)", file=sys.stderr)
        return 1

    ud = user_dirs[0]
    pf = ud / "poses_full.npy"
    vm = ud / "valid_mask.npy"
    if not pf.is_file() or not vm.is_file():
        print(f"FAIL: faltan npy en {ud}", file=sys.stderr)
        return 1

    poses = np.load(pf)
    mask = np.load(vm)
    vsum = int(mask.sum())
    ok_t = poses.shape[0] == fc == mask.shape[0]
    print(f"\nposes_full.shape[0] = {poses.shape[0]}")
    print(f"valid_mask.shape[0] = {mask.shape[0]}")
    print(f"meta frame_count     = {fc}")
    print(f"valid_mask.sum()     = {vsum}")

    if not ok_t:
        print("FAIL: T no alinea frame_count / valid_mask", file=sys.stderr)
        return 1
    if vsum <= 0:
        print("FAIL: valid_mask.sum() debe ser > 0", file=sys.stderr)
        return 1
    if not meta.get("iamodel_session_close_export"):
        print("WARN: iamodel_session_close_export no es true en meta", file=sys.stderr)

    print("\nPASS: session-close output OK (T alineado, valid_mask>0, un user_*)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
