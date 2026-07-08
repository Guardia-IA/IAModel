"""Rutas de artefactos para la campaña (separadas del entrenamiento histórico)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

CAMPAIGN_ROOT = Path(__file__).resolve().parent
CLASS_MAPS_DIR = CAMPAIGN_ROOT / "class_maps"
ARTIFACTS_ROOT = CAMPAIGN_ROOT / "artifacts"
CONFIG_PATH = CAMPAIGN_ROOT / "campaign_config.json"
IMPROVE_CONFIG_PATH = CAMPAIGN_ROOT / "campaign_config_improve.json"


def load_campaign_config(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path) if path else CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_merged_campaign_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Carga config de mejora y fusiona aug_profiles (etc.) desde `extends`."""
    if path is None:
        return load_campaign_config()
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    extends = cfg.get("extends")
    if extends:
        base_path = p.parent / str(extends) if not Path(str(extends)).is_absolute() else Path(extends)
        base = load_campaign_config(base_path)
        for key in (
            "aug_profiles",
            "threshold_sweep",
            "ensemble",
            "single_user_only",
            "robbery_class",
            "interactions_per_day",
            "target_recall_pct",
            "target_fp_rate_pct",
            "experiment_ids",
            "binary_experiment_ids",
            "multiclass_experiment_ids",
        ):
            cfg.setdefault(key, base.get(key))
    return cfg


def artifacts_root(run_id: Optional[str] = None) -> Path:
    if run_id:
        return ARTIFACTS_ROOT / "runs" / str(run_id)
    return ARTIFACTS_ROOT


def cell_dir(cell_id: str, sub: str = "", run_id: Optional[str] = None) -> Path:
    root = artifacts_root(run_id)
    base = root / sub / cell_id if sub else root / cell_id
    base.mkdir(parents=True, exist_ok=True)
    return base


def cell_artifacts(cell_id: str, run_id: Optional[str] = None) -> Dict[str, Path]:
    root = artifacts_root(run_id)
    return {
        "cell_id": cell_id,
        "run_id": run_id,
        "artifacts_root": root,
        "plans_dir": root / "plans" / cell_id,
        "models_dir": root / "models" / cell_id,
        "splits_dir": root / "splits" / cell_id,
        "reports_dir": root / "reports" / cell_id,
        "fp_clips_dir": root / "fp_clips" / cell_id,
        "logs_dir": root / "logs",
    }


def ensure_cell_dirs(cell_id: str, run_id: Optional[str] = None) -> Dict[str, Path]:
    arts = cell_artifacts(cell_id, run_id=run_id)
    for key in ("plans_dir", "models_dir", "splits_dir", "reports_dir", "fp_clips_dir", "logs_dir"):
        arts[key].mkdir(parents=True, exist_ok=True)
    return arts


def training_plan_path(cell_id: str, run_id: Optional[str] = None) -> Path:
    return cell_artifacts(cell_id, run_id=run_id)["plans_dir"] / "training_plan.json"


def category_aug_path(cell_id: str, run_id: Optional[str] = None) -> Path:
    return cell_artifacts(cell_id, run_id=run_id)["plans_dir"] / "config_category_augmentation.json"


def mass_augment_config_path(cell_id: str, run_id: Optional[str] = None) -> Path:
    return cell_artifacts(cell_id, run_id=run_id)["plans_dir"] / "config_mass_augmentation.json"


def hard_negative_manifest_path(cell_id: str, run_id: Optional[str] = None) -> Path:
    return cell_artifacts(cell_id, run_id=run_id)["plans_dir"] / "hard_negative_uids.json"


def current_run_pointer_path() -> Path:
    return ARTIFACTS_ROOT / "runs" / ".current_run"


def read_current_run_id() -> Optional[str]:
    p = current_run_pointer_path()
    if not p.is_file():
        return None
    rid = p.read_text(encoding="utf-8").strip()
    return rid or None


def write_current_run_id(run_id: str) -> Path:
    p = current_run_pointer_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(run_id).strip() + "\n", encoding="utf-8")
    return p


def new_run_id(prefix: str = "campaign") -> str:
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def run_logs_dir(run_id: str) -> Path:
    d = artifacts_root(run_id) / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_meta_path(run_id: str) -> Path:
    return artifacts_root(run_id) / "run_meta.json"


def class_map_path(map_id: str) -> Path:
    p = CLASS_MAPS_DIR / f"map_{map_id}.json"
    if not p.is_file():
        p = CLASS_MAPS_DIR / f"{map_id}.json"
    if not p.is_file():
        raise FileNotFoundError(f"No hay class map para id={map_id!r} en {CLASS_MAPS_DIR}")
    return p


DEFAULT_MASS_CELLS = (
    "mc_full",
    "mc_filtered",
    "bin_full",
    "bin_filtered",
    "bin_full_hardened",
    "bin_filtered_hardened",
)


def resolve_mass_cells(config: Dict[str, Any], cell_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Celdas del pipeline mass_augment (preflight_mass_augment / evaluate_mass_augment)."""
    ma = config.get("mass_augment") or {}
    default_ids = ma.get("cells") or list(DEFAULT_MASS_CELLS)
    ids = cell_ids or default_ids
    return filter_cells(config, list(ids))


def filter_cells(
    config: Dict[str, Any],
    cell_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    cells = list(config.get("cells", []))
    if cell_ids:
        wanted = set(cell_ids)
        cells = [c for c in cells if c["id"] in wanted]
    return cells


def resolve_experiment_ids(spec: Any) -> List[int]:
    """
    Resuelve experiment_ids del config.
    spec puede ser lista [6, 14, ...] o la cadena \"all\" (todos los EXPERIMENTS de model_config).
    """
    if spec is None:
        return []
    if spec == "all" or (isinstance(spec, str) and spec.strip().lower() == "all"):
        from model_config import EXPERIMENTS

        return list(range(1, len(EXPERIMENTS) + 1))
    if isinstance(spec, (list, tuple)):
        if len(spec) == 1 and str(spec[0]).strip().lower() == "all":
            from model_config import EXPERIMENTS

            return list(range(1, len(EXPERIMENTS) + 1))
        return [int(x) for x in spec]
    raise ValueError(f"experiment_ids inválido: {spec!r} (usa lista de enteros o \"all\")")


def _config_block(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    block = config.get(key)
    return block if isinstance(block, dict) else {}


def task_experiment_id_spec(cell: Dict[str, Any], config: Dict[str, Any]) -> Any:
    """Spec de experiment_ids según task (binary/multiclass).

    Prioridad: mass_augment → raíz del config → learning_curve → experiment_ids global.
    """
    task = str(cell.get("task") or "")
    id_key = "binary_experiment_ids" if task == "binary" else "multiclass_experiment_ids"
    sources = (
        _config_block(config, "mass_augment"),
        config,
        _config_block(config, "learning_curve"),
    )
    for source in sources:
        spec = source.get(id_key)
        if spec is not None:
            return spec
    if task == "binary":
        for source in sources:
            legacy = source.get("experiment_ids")
            if legacy is not None:
                return legacy
        return [6, 14]
    return config.get("experiment_ids") or "all"


def experiment_ids_for_cell(cell: Dict[str, Any], config: Dict[str, Any]) -> List[int]:
    """Lista de exp_id (1-based) para entrenar/evaluar la celda."""
    return resolve_experiment_ids(task_experiment_id_spec(cell, config))


def master_reports_dir(run_id: Optional[str] = None) -> Path:
    d = artifacts_root(run_id) / "reports" / "_master"
    d.mkdir(parents=True, exist_ok=True)
    return d
