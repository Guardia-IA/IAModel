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
        hint = meta.get("skip_reason") or meta.get("export_skip") or meta.get("reject_reason")
        if hint:
            errors.append(f"sin_export: {hint}")
        elif meta.get("single_user_only") and int(meta.get("track_count") or 0) > 1:
            errors.append("sin_export: single_user_only y varios tracks")
        else:
            errors.append(
                "sin carpetas user_* (multi-persona, decode vacío o filtros de calidad en pose_core)"
            )
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
    ffprobe_t = meta.get("ffprobe_nb_read_frames")
    fc = int(meta.get("frame_count") or 0)
    if ffprobe_t is not None and int(ffprobe_t) > 0 and fc != int(ffprobe_t):
        errors.append(
            f"frame_count={fc} != ffprobe_nb_read_frames={ffprobe_t} "
            "(gold N2: T offline debe igualar C1/ffprobe clip_buffer)"
        )
    for ud in user_dirs:
        pf = ud / "poses_full.npy"
        if pf.is_file():
            t_user = int(np.load(pf, mmap_mode="r").shape[0])
            if fc > 0 and t_user != fc:
                errors.append(f"{ud.name}: poses_full T={t_user} != meta.frame_count={fc}")
    return errors
