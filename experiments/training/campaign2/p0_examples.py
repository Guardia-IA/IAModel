"""Ejemplos legacy filtrados por manifest Campaign2 (split / uid)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

TRAINING_DIR = Path(__file__).resolve().parent.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from dataclasses import replace

from train_model_operations import PoseExample, _example_uid, collect_examples  # type: ignore


def map_examples_to_checkpoint_labels(
    examples: List[PoseExample],
    label_to_idx: Dict[int, int],
    *,
    robbery_class: int = 6,
) -> List[PoseExample]:
    """
    Modelos binarios (#41/#48): label_to_idx suele ser {0, 6}.
    Categorías fuera del checkpoint (p. ej. 12) se mapean a la clase negativa del .pt.
    """
    allowed = set(int(k) for k in label_to_idx.keys())
    if not allowed:
        return list(examples)
    neg_keys = sorted(k for k in allowed if int(k) != int(robbery_class))
    neg_label = int(neg_keys[0]) if neg_keys else int(min(allowed))
    rob_label = int(robbery_class) if int(robbery_class) in allowed else neg_label

    mapped: List[PoseExample] = []
    for ex in examples:
        raw = int(ex.label)
        if raw in allowed:
            new_lab = raw
        elif raw == int(robbery_class):
            new_lab = rob_label
        else:
            new_lab = neg_label
        if new_lab != raw:
            ex = replace(ex, label=new_lab, action_label=ex.action_label or raw)
        mapped.append(ex)
    return mapped


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
