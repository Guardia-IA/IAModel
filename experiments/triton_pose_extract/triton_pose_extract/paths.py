"""Resolución del repo kanvis-triton-pose (vendor externo)."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def default_triton_pose_repo() -> Path | None:
    env = os.environ.get("KANVIS_TRITON_POSE_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
    # IAModel/experiments/triton_pose_extract/triton_pose_extract/paths.py → Proyectos/
    here = Path(__file__).resolve()
    projects = here.parents[5] if len(here.parents) > 5 else here.parents[4]
    candidate = projects / "kanvis-triton-pose"
    if candidate.is_dir():
        return candidate
    return None


def ensure_pose_core_importable(repo: Path | None = None) -> Path:
    root = repo or default_triton_pose_repo()
    if root is None or not root.is_dir():
        raise RuntimeError(
            "No se encuentra kanvis-triton-pose. Exporta KANVIS_TRITON_POSE_ROOT "
            "o clónalo junto a GuadIA-IAModel (../kanvis-triton-pose)."
        )
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root
