"""Rutas de artefactos para la campaña (separadas del entrenamiento histórico)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

CAMPAIGN_ROOT = Path(__file__).resolve().parent
CLASS_MAPS_DIR = CAMPAIGN_ROOT / "class_maps"
ARTIFACTS_ROOT = CAMPAIGN_ROOT / "artifacts"
CONFIG_PATH = CAMPAIGN_ROOT / "campaign_config.json"


def load_campaign_config(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path) if path else CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def cell_dir(cell_id: str, sub: str = "") -> Path:
    base = ARTIFACTS_ROOT / sub / cell_id if sub else ARTIFACTS_ROOT / cell_id
    base.mkdir(parents=True, exist_ok=True)
    return base


def cell_artifacts(cell_id: str) -> Dict[str, Path]:
    root = ARTIFACTS_ROOT
    return {
        "cell_id": cell_id,
        "plans_dir": root / "plans" / cell_id,
        "models_dir": root / "models" / cell_id,
        "splits_dir": root / "splits" / cell_id,
        "reports_dir": root / "reports" / cell_id,
        "fp_clips_dir": root / "fp_clips" / cell_id,
        "logs_dir": root / "logs",
    }


def ensure_cell_dirs(cell_id: str) -> Dict[str, Path]:
    arts = cell_artifacts(cell_id)
    for key in ("plans_dir", "models_dir", "splits_dir", "reports_dir", "fp_clips_dir", "logs_dir"):
        arts[key].mkdir(parents=True, exist_ok=True)
    return arts


def training_plan_path(cell_id: str) -> Path:
    return cell_artifacts(cell_id)["plans_dir"] / "training_plan.json"


def category_aug_path(cell_id: str) -> Path:
    return cell_artifacts(cell_id)["plans_dir"] / "config_category_augmentation.json"


def class_map_path(map_id: str) -> Path:
    p = CLASS_MAPS_DIR / f"map_{map_id}.json"
    if not p.is_file():
        p = CLASS_MAPS_DIR / f"{map_id}.json"
    if not p.is_file():
        raise FileNotFoundError(f"No hay class map para id={map_id!r} en {CLASS_MAPS_DIR}")
    return p


def filter_cells(
    config: Dict[str, Any],
    cell_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    cells = list(config.get("cells", []))
    if cell_ids:
        wanted = set(cell_ids)
        cells = [c for c in cells if c["id"] in wanted]
    return cells


def master_reports_dir() -> Path:
    d = ARTIFACTS_ROOT / "reports" / "_master"
    d.mkdir(parents=True, exist_ok=True)
    return d
