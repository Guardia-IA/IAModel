#!/usr/bin/env python3
"""Opcional: volcar ventanas de 96 frames a disco como .npy + índice JSONL."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

WINDOW_DIR = Path(__file__).resolve().parent
TRAINING_DIR = WINDOW_DIR.parent
for p in (TRAINING_DIR, WINDOW_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from train_model_operations import collect_examples, get_data_result_roots, make_binary_examples  # type: ignore
from window_index import build_windows_from_examples
from window_preprocess import extract_window
from window_paths import load_window_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--task", choices=("binary", "multiclass"), default="binary")
    ap.add_argument("--max-windows", type=int, default=0, help="0 = todas")
    args = ap.parse_args()

    cfg = load_window_config()
    data_root = Path(args.data_root).resolve() if args.data_root else get_data_result_roots(None)[0]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "windows_index.jsonl"

    wl = int(cfg.get("window_len", 96))
    stride = int(cfg.get("stride", 48))
    pad_mode = cfg.get("pad_mode", "repeat_last")
    mask_policy = cfg.get("mask_policy", "keep_grid")

    examples = collect_examples(
        pose_source=cfg.get("pose_source", "full"),
        single_user_only=bool(cfg.get("single_user_only", True)),
        data_root=data_root,
    )
    if args.task == "binary":
        examples = make_binary_examples(examples, positive_classes=cfg.get("robbery_classes", [6, 14]))

    windows = build_windows_from_examples(
        examples, window_len=wl, stride=stride, pose_source=cfg.get("pose_source", "full")
    )
    if args.max_windows > 0:
        windows = windows[: args.max_windows]

    n = 0
    with open(index_path, "w", encoding="utf-8") as idx:
        for w in windows:
            poses_full = np.load(w.pose_path)
            vm = np.load(w.valid_mask_path) if w.valid_mask_path and w.valid_mask_path.exists() else None
            chunk, chunk_m, _pf = extract_window(
                poses_full, vm, w.start, w.end, window_len=wl, pad_mode=pad_mode, mask_policy=mask_policy
            )
            fname = f"w_{n:07d}_{w.clip_uid.replace('::', '_')}_s{w.start:04d}.npy"
            fpath = out_dir / fname
            np.save(fpath, chunk)
            rec = {
                "path": str(fpath),
                "label": w.label,
                "window_id": w.window_id,
                "clip_uid": w.clip_uid,
                "start": w.start,
                "end": w.end,
                "source_pose": str(w.pose_path),
            }
            if chunk_m is not None:
                mask_path = fpath.with_suffix(".mask.npy")
                np.save(mask_path, chunk_m)
                rec["valid_mask_path"] = str(mask_path)
            idx.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"Materializadas {n} ventanas en {out_dir} | índice {index_path}")


if __name__ == "__main__":
    main()