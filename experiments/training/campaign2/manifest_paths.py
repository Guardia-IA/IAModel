"""Resolución de manifest.json (modo simple vs matriz full/filtered)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from campaign2_paths import run_layout, run_meta_path


def load_run_meta(run_id: str) -> Dict[str, Any]:
    p = run_meta_path(run_id)
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def write_run_meta(run_id: str, doc: Dict[str, Any]) -> Path:
    p = run_meta_path(run_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return p


def discover_manifest_by_pose(cache_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for sub in ("full", "filtered"):
        mp = cache_dir / sub / "manifest.json"
        if mp.is_file():
            out[sub] = mp
    return out


def default_manifest_path(run_id: str) -> Optional[Path]:
    meta = load_run_meta(run_id)
    by_pose = meta.get("manifest_by_pose") or {}
    if isinstance(by_pose, dict) and by_pose:
        first = next(iter(by_pose.values()))
        return Path(str(first))
    layout = run_layout(run_id)
    root_man = layout["cache_dir"] / "manifest.json"
    if root_man.is_file():
        return root_man
    discovered = discover_manifest_by_pose(layout["cache_dir"])
    if discovered:
        return next(iter(discovered.values()))
    return None


def manifest_for_job_id(run_id: str, job_id: str) -> Optional[Path]:
    meta = load_run_meta(run_id)
    by_pose: Dict[str, str] = dict(meta.get("manifest_by_pose") or {})
    if not by_pose:
        by_pose = {k: str(v) for k, v in discover_manifest_by_pose(run_layout(run_id)["cache_dir"]).items()}
    jid = job_id.lower()
    if "_filtered" in jid or jid.endswith("_filtered"):
        p = by_pose.get("filtered")
        return Path(p) if p else None
    if "_full" in jid or jid.endswith("_full"):
        p = by_pose.get("full")
        return Path(p) if p else None
    return default_manifest_path(run_id)
