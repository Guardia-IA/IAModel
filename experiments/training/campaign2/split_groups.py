"""Agrupación de split (video / sesión / take) — reduce leakage obvio."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict


def stable_group_id(rec: Dict[str, Any], *, strategy: str = "stratified_group_hybrid") -> str:
    meta = rec.get("meta") or {}
    clip_dir = Path(str(rec.get("clip_dir") or ""))
    vs = str(meta.get("video_source") or meta.get("video") or meta.get("camera_id") or "")
    name = clip_dir.name

    if strategy in ("stratified_group_hybrid", "session_prefix"):
        prefix = name.split("_")[0] if "_" in name else name[:12]
        actor = str(meta.get("actor_id") or meta.get("take_id") or "")
        users = meta.get("users") or []
        if not actor and isinstance(users, list) and len(users) == 1:
            u0 = users[0]
            if isinstance(u0, dict):
                actor = str(u0.get("session_id") or u0.get("track_id") or u0.get("id") or "")
        raw = f"{vs}|{actor}|{prefix}"
    else:
        raw = f"{vs}|{name}"

    raw = re.sub(r"\s+", " ", raw.strip())
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
