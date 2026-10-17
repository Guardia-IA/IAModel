"""Elige un solo user_* cuando el tracker asignó varios IDs (entrenamiento single-person)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def has_complete_export(clip_root: Path) -> bool:
    if not clip_root.is_dir():
        return False
    for ud in clip_root.glob("user_*"):
        if (ud / "poses_full.npy").is_file():
            return True
    return False


def _score_user(entry: Dict[str, Any]) -> float:
    vf = float(entry.get("valid_frames") or 0)
    vp = float(entry.get("valid_pct") or 0)
    return vf * 1000.0 + vp


def prune_to_primary_user(clip_root: Path) -> Tuple[bool, Optional[str]]:
    """
    Deja solo user_{track_id} con más frames válidos (meta.users).
    Devuelve (ok, mensaje).
    """
    meta_path = clip_root / "meta.json"
    if not meta_path.is_file():
        return False, "sin meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    users: List[Dict[str, Any]] = list(meta.get("users") or [])
    user_dirs = [p for p in clip_root.glob("user_*") if p.is_dir()]
    if not user_dirs:
        return False, "sin user_*"
    if len(user_dirs) == 1 and len(users) <= 1:
        return True, None

    if users:
        best = max(users, key=_score_user)
        tid = int(best.get("track_id", 0))
        keep_name = f"user_{tid}"
    else:
        keep_name = max(user_dirs, key=lambda p: (p / "poses_full.npy").stat().st_size if (p / "poses_full.npy").is_file() else 0).name
        tid = int(keep_name.replace("user_", "") or 0)

    removed = 0
    for ud in user_dirs:
        if ud.name == keep_name:
            continue
        shutil.rmtree(ud, ignore_errors=True)
        removed += 1

    if users:
        kept = [u for u in users if int(u.get("track_id", -1)) == tid]
        meta["users"] = kept if kept else [max(users, key=_score_user)]
    meta["iamodel_primary_user_track_id"] = tid
    meta["iamodel_primary_user_pruned"] = removed
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not (clip_root / keep_name / "poses_full.npy").is_file():
        return False, f"falta {keep_name}/poses_full.npy tras poda"
    return True, None
