#!/usr/bin/env python3
"""Build or refresh core_index.json under window_campaign2/artifacts/."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TR = ROOT.parent
WC = TR / "window_campaign"
for p in (TR, WC, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from campaign2_paths import ensure_core_index, load_campaign2_config  # noqa: E402
from core_labels import build_core_index_for_examples  # type: ignore
from train_model_operations import collect_examples, make_binary_examples  # type: ignore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    args = ap.parse_args()
    cfg = load_campaign2_config()
    path = ensure_core_index(cfg)
    ex = collect_examples(data_root=Path(args.data_root).resolve(), pose_source=cfg.get("pose_source", "full"))
    ex = make_binary_examples(ex, positive_classes=cfg.get("robbery_classes", [6, 14]))
    build_core_index_for_examples(ex, path, cfg.get("robbery_classes", [6, 14]))
    print(f"[WIN2] core_index → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())