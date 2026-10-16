"""Ejemplos legacy filtrados por manifest Campaign2 (split / uid)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

TRAINING_DIR = Path(__file__).resolve().parent.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from train_model_operations import PoseExample, _example_uid, collect_examples  # type: ignore


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def uids_for_split(manifest: Dict[str, Any], split_name: str) -> Set[str]:
    out: Set[str] = set()
    for e in manifest.get("entries") or []:
        if str(e.get("split")) == split_name:
            uid = e.get("uid")
            if uid:
                out.add(str(uid))
    return out


def manifest_entries_for_split(manifest: Dict[str, Any], split_name: str) -> List[Dict[str, Any]]:
    return [e for e in (manifest.get("entries") or []) if str(e.get("split")) == split_name]


def collect_examples_for_manifest(
    data_roots: List[Path],
    manifest: Dict[str, Any],
    *,
    eval_split: str,
    pose_source: str,
    single_user_only: bool,
) -> List[PoseExample]:
    want = uids_for_split(manifest, eval_split)
    if not want:
        return []
    by_uid: Dict[str, PoseExample] = {}
    for root in data_roots:
        examples = collect_examples(
            pose_source=pose_source,
            single_user_only=single_user_only,
            data_root=str(root),
        )
        for ex in examples:
            uid = _example_uid(ex)
            if uid in want and uid not in by_uid:
                by_uid[uid] = ex
    missing = want - set(by_uid)
    found = want & set(by_uid)
    if not found:
        sample = sorted(want)[:5]
        raise RuntimeError(
            f"P0: ningún uid del split en data_roots (ej. esperados {sample}). "
            "¿Mismas raíces que en preflight?"
        )
    if missing:
        sample = sorted(missing)[:5]
        print(
            f"P0 WARN: {len(missing)} uids del split no aparecen con single_user_only={single_user_only} "
            f"(ej. {sample})",
            flush=True,
        )
    order = [
        e["uid"]
        for e in manifest.get("entries") or []
        if str(e.get("uid")) in found and str(e.get("split")) == eval_split
    ]
    return [by_uid[str(u)] for u in order if str(u) in by_uid]
