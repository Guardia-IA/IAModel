"""Rutas de artefactos window_campaign (separado de campaign/)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

WINDOW_ROOT = Path(__file__).resolve().parent
WINDOW_DIR = WINDOW_ROOT
CONFIG_PATH = WINDOW_ROOT / "window_config.json"
ARTIFACTS_ROOT = WINDOW_ROOT / "artifacts"


def load_window_config(path: Optional[Path] = None) -> Dict[str, Any]:
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


def plans_dir(cell_id: str = "win_bin_full") -> Path:
    d = ARTIFACTS_ROOT / "plans" / cell_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir(cell_id: str = "win_bin_full") -> Path:
    d = ARTIFACTS_ROOT / "models" / cell_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def window_plan_path(cell_id: str = "win_bin_full") -> Path:
    return plans_dir(cell_id) / "window_training_plan.json"