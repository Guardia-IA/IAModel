"""Resolución del repo kanvis-triton-pose (vendor externo)."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _repo_candidates() -> list[Path]:
    here = Path(__file__).resolve()
    home = Path.home()
    iamodel = here.parents[3] if len(here.parents) > 3 else here
    roots = [
        home / "kanvis-triton-pose",
        home / "dev" / "kanvis-triton-pose",
        home / "dev" / "Proyectos" / "kanvis-triton-pose",
        iamodel.parent / "kanvis-triton-pose",
        iamodel / "vendor" / "kanvis-triton-pose",
    ]
    if len(here.parents) > 5:
        roots.append(here.parents[5] / "kanvis-triton-pose")
    seen: set[str] = set()
    out: list[Path] = []
    for p in roots:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def default_triton_pose_repo() -> Path | None:
    env = os.environ.get("KANVIS_TRITON_POSE_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
    for candidate in _repo_candidates():
        if candidate.is_dir() and (candidate / "pose_core").is_dir():
            return candidate
    return None


def ensure_pose_core_importable(repo: Path | None = None) -> Path:
    root = repo or default_triton_pose_repo()
    if root is None or not root.is_dir():
        tried = ", ".join(str(p) for p in _repo_candidates()[:4])
        raise RuntimeError(
            "No se encuentra kanvis-triton-pose (pose_core). "
            "Exporta KANVIS_TRITON_POSE_ROOT=/ruta/al/repo o clónalo. "
            f"Probado: {tried} …"
        )
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root
