"""Split manifest paths rooted in window_campaign2/."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys

from campaign2_paths import CAMPAIGN2_ROOT

TRAINING_DIR = CAMPAIGN2_ROOT.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from train_model_operations import (  # type: ignore
    PoseExample,
    SEED,
    split_examples_by_uid_manifest,
    split_examples_stratified_by_uid,
)
from window_split_policy import (  # type: ignore
    certify_disjoint_splits,
    load_split_manifest,
    write_split_manifest,
)


def split_manifest_path(cfg: Dict[str, Any], cell_id: str | None = None) -> Path:
    split = cfg.get("split") or {}
    rel = split.get(
        "manifest_path",
        f"artifacts/plans/{cell_id or cfg.get('default_cell_id')}_split_manifest.json",
    )
    p = Path(rel)
    return p if p.is_absolute() else CAMPAIGN2_ROOT / p


def split_clips_from_config(
    examples: List[PoseExample],
    cfg: Dict[str, Any],
    *,
    cell_id: str | None = None,
    force_new_split: bool = False,
) -> Tuple[List[PoseExample], List[PoseExample], List[PoseExample], Path, bool]:
    split = cfg.get("split") or {}
    path = split_manifest_path(cfg, cell_id=cell_id)
    use_frozen = bool(split.get("use_frozen_manifest", True)) and not force_new_split
    if use_frozen and path.is_file():
        data = load_split_manifest(path)
        su = data.get("split_uids")
        if not su:
            raise ValueError(f"Manifiesto sin split_uids: {path}")
        tr, va, te = split_examples_by_uid_manifest(examples, su)
        certify_disjoint_splits(tr, va, te, label="(manifiesto congelado win2)")
        return tr, va, te, path, True

    tr, va, te = split_examples_stratified_by_uid(
        examples,
        seed=SEED,
        train_ratio=float(split.get("train_ratio", 0.8)),
        val_ratio=float(split.get("val_ratio", 0.1)),
    )
    certify_disjoint_splits(tr, va, te, label="(split nuevo win2)")
    return tr, va, te, path, False