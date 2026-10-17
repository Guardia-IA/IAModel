"""Validación mínima de artefactos IAModel."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def validate_clip_dir(clip_root: Path) -> list[str]:
    errors: list[str] = []
    meta_path = clip_root / "meta.json"
    if not meta_path.is_file():
        errors.append("falta meta.json")
        return errors
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    users = meta.get("users") or []
    user_dirs = list(clip_root.glob("user_*"))
    if not user_dirs:
        errors.append("sin carpetas user_*")
    for ud in user_dirs:
        pf = ud / "poses_full.npy"
        vm = ud / "valid_mask.npy"
        if not pf.is_file():
            errors.append(f"{ud.name}: falta poses_full.npy")
            continue
        arr = np.load(pf, mmap_mode="r")
        if arr.ndim != 3 or arr.shape[1] != 8 or arr.shape[2] != 2:
            errors.append(f"{ud.name}: poses_full shape {arr.shape} (esperado T,8,2)")
        if vm.is_file():
            vm_arr = np.load(vm, mmap_mode="r")
            if vm_arr.ndim != 1 or vm_arr.shape[0] != arr.shape[0]:
                errors.append(f"{ud.name}: valid_mask incompatible con T={arr.shape[0]}")
    if users and len(user_dirs) != len(users):
        errors.append(f"meta.users={len(users)} vs dirs={len(user_dirs)}")
    return errors
