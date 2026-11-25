"""Expansión sintética por ventana (manifest validate_npy a nivel clip)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from train_model_operations import (  # type: ignore
    PoseExample,
    resolve_manifest_json_for_example,
    specs_from_validate_manifest_payload,
)

from window_index import WindowExample


def _window_copy(w: WindowExample, forced_ops: Optional[List[Dict[str, Any]]]) -> WindowExample:
    return WindowExample(
        pose_path=w.pose_path,
        valid_mask_path=w.valid_mask_path,
        label=w.label,
        clip_name=w.clip_name,
        category_str=w.category_str,
        track_id=w.track_id,
        action_label=w.action_label,
        start=w.start,
        end=w.end,
        window_id=w.window_id + ("::grid" if forced_ops == [] else "::syn"),
        clip_uid=w.clip_uid,
        forced_ops=forced_ops,
    )


def _clip_pose_example_from_window(w: WindowExample) -> PoseExample:
    return PoseExample(
        pose_path=w.pose_path,
        label=w.label,
        track_id=w.track_id,
        clip_name=w.clip_name,
        category_str=w.category_str,
        valid_mask_path=w.valid_mask_path,
        action_label=w.action_label,
        forced_ops=getattr(w, "forced_ops", None),
    )


def expand_windows_with_manifest_extra_views(
    windows: List[WindowExample],
    mdir: Path,
    variant_set: str,
    extra_views_per_window: int,
) -> List[WindowExample]:
    """
    Por cada ventana de un clip con manifest, añade filas identidad + hasta N variantes.
    Copia la misma ventana [start:end] con distintas forced_ops (sintético por chunk).
    """
    if extra_views_per_window <= 0:
        return windows
    mdir = Path(mdir)
    if not mdir.is_dir():
        print("[WINDOW-EXPAND] manifest_cache_dir inválido; sin expansión.")
        return windows
    n_req = int(extra_views_per_window)
    manifest_by_clip: Dict[str, List[Dict[str, Any]]] = {}
    skipped = 0
    for w in windows:
        if w.clip_uid in manifest_by_clip:
            continue
        ex = _clip_pose_example_from_window(w)
        p = resolve_manifest_json_for_example(mdir, ex)
        if p is None:
            manifest_by_clip[w.clip_uid] = []
            skipped += 1
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            specs = specs_from_validate_manifest_payload(data, variant_set=variant_set)
            manifest_by_clip[w.clip_uid] = specs or []
        except Exception:
            manifest_by_clip[w.clip_uid] = []
            skipped += 1

    out: List[WindowExample] = []
    for w in windows:
        if getattr(w, "forced_ops", None) is not None:
            out.append(w)
            continue
        specs = manifest_by_clip.get(w.clip_uid) or []
        if not specs:
            out.append(w)
            continue
        out.append(_window_copy(w, []))
        avail = max(0, len(specs) - 1)
        n_extra = min(n_req, avail)
        for j in range(n_extra):
            ops = list(specs[1 + j].get("ops", []) or [])
            out.append(_window_copy(w, ops))

    print(
        f"[WINDOW-EXPAND] ventanas: {len(windows)} -> {len(out)} | "
        f"extra_views={n_req} | clips sin manifest~{skipped}"
    )
    return out