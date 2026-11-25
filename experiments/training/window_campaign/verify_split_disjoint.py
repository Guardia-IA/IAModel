#!/usr/bin/env python3
"""Comprueba que train / val / test no comparten clips (UID ni ruta .npy)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WINDOW_DIR = Path(__file__).resolve().parent
TRAINING_DIR = WINDOW_DIR.parent
for p in (TRAINING_DIR, WINDOW_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from train_model_operations import (  # type: ignore
    PoseExample,
    _example_uid,
    collect_examples,
    make_binary_examples,
    split_examples_by_uid_manifest,
    verify_split_uid_disjoint,
)
from window_paths import load_window_config  # noqa: E402
from window_split_policy import certify_disjoint_splits, split_manifest_path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--manifest", type=str, default=None)
    ap.add_argument("--cell-id", type=str, default=None)
    args = ap.parse_args()

    cfg = load_window_config()
    cell_id = args.cell_id or cfg.get("default_cell_id")
    mpath = Path(args.manifest) if args.manifest else split_manifest_path(cfg, cell_id=cell_id)
    if not mpath.is_file():
        print(f"ERROR: no existe manifiesto {mpath}")
        return 1

    with open(mpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    su = data["split_uids"]
    verify_split_uid_disjoint(su)

    # intra-manifest
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = set(su.get(a, [])) & set(su.get(b, []))
        if inter:
            print(f"FAIL manifest: {a} ∩ {b} = {len(inter)} UIDs")
            return 1
    print(f"OK manifiesto: UIDs disjuntos train={len(su.get('train', []))} val={len(su.get('val', []))} test={len(su.get('test', []))}")

    if args.data_root:
        from train_model_operations import get_data_result_roots

        root = Path(args.data_root).resolve()
        cf = cfg.get("clip_filters") or {}
        ex = collect_examples(
            pose_source=cfg.get("pose_source", "full"),
            single_user_only=bool(cfg.get("single_user_only", True)),
            data_root=root,
            min_clip_seconds=float(cf.get("min_clip_seconds", 0)),
        )
        ex = make_binary_examples(ex, positive_classes=cfg.get("robbery_classes", [6, 14]))
        train, val, test = split_examples_by_uid_manifest(ex, su)
        certify_disjoint_splits(train, val, test, label="(verify vs data_root)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())