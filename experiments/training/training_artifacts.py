"""
Rutas de artefactos separadas multiclase vs binario (plan, augment, modelos, splits, informes).

Multiclase no se sobrescribe al entrenar/evaluar binario (6 vs resto).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

TRAINING_ROOT = Path(__file__).resolve().parent

# Multiclase (por defecto histórico)
TRAINING_PLAN_MULTICLASS_PATH = TRAINING_ROOT / "training_plan.json"
CATEGORY_AUGMENTATION_MULTICLASS_PATH = TRAINING_ROOT / "config_category_augmentation.json"
MODELS_MULTICLASS_DIR = TRAINING_ROOT / "models-operation"
MODELS_MULTICLASS_SINGLE_DIR = TRAINING_ROOT / "models-operation-single"
SPLITS_MULTICLASS_DIR = TRAINING_ROOT / "splits"
REPORTS_MULTICLASS_DIR = TRAINING_ROOT / "reports_multiclass"

# Binario (6 vs resto) — rutas independientes
TRAINING_PLAN_BINARY_PATH = TRAINING_ROOT / "training_plan_binary.json"
CATEGORY_AUGMENTATION_BINARY_PATH = TRAINING_ROOT / "config_category_augmentation_binary.json"
MODELS_BINARY_DIR = TRAINING_ROOT / "models-operation-binary"
MODELS_BINARY_SINGLE_DIR = TRAINING_ROOT / "models-operation-single-binary"
SPLITS_BINARY_DIR = TRAINING_ROOT / "splits_binary"
REPORTS_BINARY_DIR = TRAINING_ROOT / "reports_binary"


def default_artifacts(task: str, *, single_user_only: bool = False) -> Dict[str, Path]:
    task = str(task).lower()
    if task == "binary":
        return {
            "task": "binary",
            "training_plan": TRAINING_PLAN_BINARY_PATH,
            "category_aug_config": CATEGORY_AUGMENTATION_BINARY_PATH,
            "models_dir": MODELS_BINARY_SINGLE_DIR if single_user_only else MODELS_BINARY_DIR,
            "splits_dir": SPLITS_BINARY_DIR,
            "reports_dir": REPORTS_BINARY_DIR,
        }
    return {
        "task": "multiclass",
        "training_plan": TRAINING_PLAN_MULTICLASS_PATH,
        "category_aug_config": CATEGORY_AUGMENTATION_MULTICLASS_PATH,
        "models_dir": MODELS_MULTICLASS_SINGLE_DIR if single_user_only else MODELS_MULTICLASS_DIR,
        "splits_dir": SPLITS_MULTICLASS_DIR,
        "reports_dir": REPORTS_MULTICLASS_DIR,
    }


def resolve_artifacts(
    task: str,
    *,
    single_user_only: bool = False,
    training_plan: Optional[str | Path] = None,
    category_aug_config: Optional[str | Path] = None,
    models_dir: Optional[str | Path] = None,
    splits_dir: Optional[str | Path] = None,
    reports_dir: Optional[str | Path] = None,
    mkdir: bool = True,
) -> Dict[str, Path]:
    """Devuelve rutas finales; los overrides explícitos prevalecen sobre defaults por task."""
    out = default_artifacts(task, single_user_only=single_user_only)
    if training_plan is not None:
        out["training_plan"] = Path(training_plan)
    if category_aug_config is not None:
        out["category_aug_config"] = Path(category_aug_config)
    if models_dir is not None:
        out["models_dir"] = Path(models_dir)
    if splits_dir is not None:
        out["splits_dir"] = Path(splits_dir)
    if reports_dir is not None:
        out["reports_dir"] = Path(reports_dir)
    if mkdir:
        for key in ("models_dir", "splits_dir", "reports_dir"):
            out[key].mkdir(parents=True, exist_ok=True)
    return out


def print_artifact_banner(artifacts: Dict[str, Path], title: str = "Artefactos") -> None:
    print(f"\n=== {title} ({artifacts.get('task', '?')}) ===")
    for key in ("training_plan", "category_aug_config", "models_dir", "splits_dir", "reports_dir"):
        if key in artifacts:
            print(f"  {key}: {artifacts[key]}")
