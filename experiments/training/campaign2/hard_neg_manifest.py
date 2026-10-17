"""Manifest P3: hard negatives (clases difíciles + opcional lista externa)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set


def build_hard_neg_manifest(
    records: List[Dict[str, Any]],
    *,
    hard_neg_classes: List[int],
    robbery_class: int = 6,
    extra_uids: Set[str] | None = None,
) -> Dict[str, Any]:
    hard_set = {int(x) for x in hard_neg_classes}
    rows: List[Dict[str, Any]] = []
    uids: Set[str] = set(extra_uids or ())
    for rec in records:
        lb = int(rec.get("label", -1))
        uid = str(rec.get("uid") or "")
        if not uid:
            continue
        if lb in hard_set and lb != int(robbery_class):
            rows.append(
                {
                    "uid": uid,
                    "label": lb,
                    "clip_dir": rec.get("clip_dir"),
                    "source": "inventory_class",
                }
            )
            uids.add(uid)
    return {
        "version": 1,
        "robbery_class": int(robbery_class),
        "hard_neg_classes": sorted(hard_set),
        "n_clips": len(rows),
        "uids": sorted(uids),
        "entries": rows,
    }


def load_hard_neg_uids(path: Path) -> Set[str]:
    if not path.is_file():
        return set()
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {str(u) for u in (doc.get("uids") or [])}


def write_hard_neg_manifest(path: Path, doc: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def merge_external_manifest(doc: Dict[str, Any], external_path: Path) -> Dict[str, Any]:
    if not external_path.is_file():
        return doc
    ext = json.loads(external_path.read_text(encoding="utf-8"))
    uids = set(doc.get("uids") or [])
    entries = list(doc.get("entries") or [])
    for row in ext.get("entries") or []:
        uid = str(row.get("uid") or "")
        if not uid or uid in uids:
            continue
        entries.append({**row, "source": row.get("source", "external")})
        uids.add(uid)
    doc = dict(doc)
    doc["entries"] = entries
    doc["uids"] = sorted(uids)
    doc["n_clips"] = len(entries)
    doc["external_merged"] = str(external_path)
    return doc
