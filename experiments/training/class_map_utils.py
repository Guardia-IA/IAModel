"""
Utilidades para remapear / excluir categorías en entrenamiento y evaluación.
Usado por campaign/ y train_model_operations (hook mínimo vía training_plan.class_map).
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from train_model_operations import PoseExample  # pragma: no cover


def load_class_map(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "id" not in data:
        raise ValueError(f"class_map sin campo 'id': {path}")
    return data


def _folder_category(ex: "PoseExample") -> int:
    try:
        from .train_model_operations import _example_folder_category
    except ImportError:
        from train_model_operations import _example_folder_category  # type: ignore
    return int(_example_folder_category(ex))


def _copy_with_label(ex: "PoseExample", label: int) -> "PoseExample":
    try:
        from .train_model_operations import PoseExample, _example_action_label
    except ImportError:
        from train_model_operations import PoseExample, _example_action_label  # type: ignore
    return PoseExample(
        pose_path=ex.pose_path,
        label=int(label),
        track_id=ex.track_id,
        clip_name=ex.clip_name,
        category_str=ex.category_str,
        valid_mask_path=getattr(ex, "valid_mask_path", None),
        users_in_clip=getattr(ex, "users_in_clip", 1),
        action_label=_example_action_label(ex),
        forced_ops=getattr(ex, "forced_ops", None),
    )


def categories_below_clip_threshold(
    folder_scan: Dict[int, Dict[str, int]],
    threshold: int,
    *,
    count_key: str = "clips",
) -> List[int]:
    """Categorías con menos de `threshold` clips listos (p. ej. meta.json en data_result)."""
    thr = max(1, int(threshold))
    out: List[int] = []
    for cat, info in sorted(folder_scan.items(), key=lambda x: int(x[0])):
        n = int((info or {}).get(count_key, 0) or 0)
        if n < thr:
            out.append(int(cat))
    return out


def resolve_class_map_spec(
    class_map_spec: Dict[str, Any],
    folder_scan: Optional[Dict[int, Dict[str, int]]] = None,
) -> Dict[str, Any]:
    """
    Fusiona exclude estático + exclude_below_clip_count (dinámico según inventario).
    Devuelve copia del spec con 'exclude' ya resuelto (lista ordenada).
    """
    spec = deepcopy(class_map_spec or {})
    exclude = {int(x) for x in spec.get("exclude", [])}
    threshold = spec.get("exclude_below_clip_count")
    if threshold is not None and folder_scan:
        for cat in categories_below_clip_threshold(folder_scan, int(threshold)):
            exclude.add(cat)
    spec["exclude"] = sorted(exclude)
    if threshold is not None:
        spec["exclude_below_clip_count"] = int(threshold)
    return spec


def apply_class_map_spec(
    examples: List["PoseExample"],
    class_map_spec: Dict[str, Any],
) -> List["PoseExample"]:
    """
    - exclude: lista de categorías de carpeta a eliminar del dataset.
    - exclude_below_clip_count: resolver antes con resolve_class_map_spec + folder_scan.
    - remap: dict orig -> nueva etiqueta de entrenamiento (solo multiclass).
    """
    spec = class_map_spec or {}
    exclude = {int(x) for x in spec.get("exclude", [])}
    remap_raw = spec.get("remap") or {}
    remap = {int(k): int(v) for k, v in remap_raw.items()}

    out: List["PoseExample"] = []
    for ex in examples:
        cat = _folder_category(ex)
        if cat in exclude:
            continue
        new_label = int(remap.get(cat, cat))
        if new_label != ex.label:
            out.append(_copy_with_label(ex, new_label))
        else:
            out.append(ex)
    return out


def plan_class_map_block(
    class_map_spec: Dict[str, Any],
    folder_scan: Optional[Dict[int, Dict[str, int]]] = None,
) -> Dict[str, Any]:
    """Subconjunto serializable para training_plan.json (exclude ya resuelto)."""
    resolved = resolve_class_map_spec(class_map_spec, folder_scan)
    block: Dict[str, Any] = {
        "id": resolved.get("id"),
        "description": resolved.get("description"),
        "exclude": [int(x) for x in resolved.get("exclude", [])],
        "remap": {str(k): int(v) for k, v in (resolved.get("remap") or {}).items()},
        "robbery_class": int(resolved.get("robbery_class", 6)),
    }
    if resolved.get("exclude_below_clip_count") is not None:
        block["exclude_below_clip_count"] = int(resolved["exclude_below_clip_count"])
    return block


def adjust_augment_for_fp_hardened(
    proposed: Dict[int, int],
    *,
    robbery_class: int = 6,
    boost_factor: float = 1.35,
) -> Dict[int, int]:
    """Sube augment en negativos; mantiene robo moderado."""
    out: Dict[int, int] = {}
    for cat, count in proposed.items():
        c = int(cat)
        if c == int(robbery_class):
            out[c] = max(0, int(round(int(count) * 0.85)))
        else:
            out[c] = int(round(int(count) * boost_factor))
    return out
