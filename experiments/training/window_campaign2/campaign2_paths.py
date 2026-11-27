"""Artifact paths for window_campaign2 (separate from window_campaign/)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

CAMPAIGN2_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = CAMPAIGN2_ROOT / "campaign2_config.json"
ARTIFACTS_ROOT = CAMPAIGN2_ROOT / "artifacts"
WINDOW_V1_ROOT = CAMPAIGN2_ROOT.parent / "window_campaign"


def load_campaign2_config(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path) if path else CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    ext = cfg.get("extends_campaign_config")
    if ext:
        cp = p.parent / ext if not Path(str(ext)).is_absolute() else Path(ext)
        if cp.is_file():
            with open(cp, "r", encoding="utf-8") as f:
                cfg["_campaign_config"] = json.load(f)
    return cfg


def plans_dir(cell_id: str) -> Path:
    d = ARTIFACTS_ROOT / "plans" / cell_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir(cell_id: str) -> Path:
    d = ARTIFACTS_ROOT / "models" / cell_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def reports_dir() -> Path:
    d = ARTIFACTS_ROOT / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir() -> Path:
    d = ARTIFACTS_ROOT / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_core_index(cfg: Dict[str, Any]) -> Path:
    pl = cfg.get("positive_labeling") or {}
    rel = str(pl.get("core_index_path", "artifacts/core_index.json"))
    out = CAMPAIGN2_ROOT / rel if not Path(rel).is_absolute() else Path(rel)
    if out.is_file():
        return out
    src_rel = pl.get("copy_core_index_from_v1")
    if src_rel:
        src = CAMPAIGN2_ROOT / src_rel if not Path(str(src_rel)).is_absolute() else Path(src_rel)
        if src.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            return out
    return out


def resolve_checkpoint_roots(cfg: Dict[str, Any]) -> Dict[str, Path]:
    p0 = cfg.get("phase0") or {}
    cell = str(cfg.get("default_cell_id", "win2_bin_native_baseline"))
    roots: Dict[str, Path] = {
        "campaign2": models_dir(cell),
        "window_v1": (CAMPAIGN2_ROOT / p0["window_v1_checkpoint_dir"]).resolve()
        if p0.get("window_v1_checkpoint_dir")
        else WINDOW_V1_ROOT / "artifacts" / "models" / "win_bin_full_baseline",
    }
    if p0.get("full_clip_checkpoint_dir"):
        roots["full_clip"] = Path(p0["full_clip_checkpoint_dir"]).resolve()
    return roots