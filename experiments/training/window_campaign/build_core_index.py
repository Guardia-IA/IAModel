#!/usr/bin/env python3
"""Build core_index.json for staged positives (categories 6, 14)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

WINDOW_DIR = Path(__file__).resolve().parent
TRAINING_DIR = WINDOW_DIR.parent
for p in (TRAINING_DIR, WINDOW_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core_labels import build_core_index_for_examples  # noqa: E402
from train_model_operations import collect_examples, get_data_result_roots  # noqa: E402
from window_paths import ARTIFACTS_ROOT, load_window_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    cfg = load_window_config()
    data_root = Path(args.data_root).resolve() if args.data_root else get_data_result_roots(None)[0]
    out = Path(args.out) if args.out else ARTIFACTS_ROOT / "core_index.json"
    ex = collect_examples(
        pose_source=cfg.get("pose_source", "full"),
        single_user_only=bool(cfg.get("single_user_only", True)),
        data_root=data_root,
    )
    rob = cfg.get("robbery_classes", [6, 14])
    clips = build_core_index_for_examples(ex, out, rob)
    print(f"Wrote {len(clips)} core entries -> {out}")


if __name__ == "__main__":
    main()