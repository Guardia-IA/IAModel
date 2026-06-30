from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, get_worker_info

# Soportar ejecución como módulo (-m training.train_model) y como script (python training/train_model.py)
try:
    from .model_config import (  # type: ignore[attr-defined]
        EXPERIMENTS,
        DATA_RESULT_ROOT,
        SPLIT_RATIO_TRAIN,
        SPLIT_RATIO_VAL,
        suggest_split_ratios,
        MIN_CLIP_SECONDS,
        MIN_VALID_FRAMES,
        MIN_VALID_PCT,
        MAX_OCCLUSION_RATIO,
        SEED,
        AUGMENT_PROB,
        AUGMENT_MAX_OPS,
        AUGMENT_SPEED_FACTOR_LO,
        AUGMENT_SPEED_FACTOR_HI,
        MAX_DETERMINISTIC_VARIANTS,
        TRAIN_DETERMINISTIC_PROB,
        AUGMENT_PROFILE_DEFAULT,
        MANIFEST_VARIANT_SET_DEFAULT,
        EXTRA_MANIFEST_VIEWS_PER_CLIP_DEFAULT,
        CATEGORY_AUGMENTATION_CONFIG_PATH,
        ROBBERY_CLASS,
        PREFLIGHT_MIN_ROBBERY_TRAIN_ROWS,
        PREFLIGHT_MIN_NEGATIVE_TRAIN_ROWS,
        PREFLIGHT_NEGATIVE_TO_ROBBERY_RATIO,
        PREFLIGHT_ROBBERY_DOMINANCE_THRESHOLD,
        PREFLIGHT_ROBBERY_RARE_THRESHOLD,
        DEFAULT_BINARY_SOFTMAX_THRESHOLD,
        DEFAULT_BINARY_LOGIT_MARGIN,
        TRAINING_PLAN_PATH,
    )
    from .training_artifacts import resolve_artifacts, print_artifact_banner  # type: ignore[attr-defined]
except ImportError:
    from model_config import (  # type: ignore[attr-defined]
        DATA_RESULT_ROOT,
        EXPERIMENTS,
        SPLIT_RATIO_TRAIN,
        SPLIT_RATIO_VAL,
        suggest_split_ratios,
        MIN_CLIP_SECONDS,
        MIN_VALID_FRAMES,
        MIN_VALID_PCT,
        MAX_OCCLUSION_RATIO,
        SEED,
        AUGMENT_PROB,
        AUGMENT_MAX_OPS,
        AUGMENT_SPEED_FACTOR_LO,
        AUGMENT_SPEED_FACTOR_HI,
        MAX_DETERMINISTIC_VARIANTS,
        TRAIN_DETERMINISTIC_PROB,
        AUGMENT_PROFILE_DEFAULT,
        MANIFEST_VARIANT_SET_DEFAULT,
        EXTRA_MANIFEST_VIEWS_PER_CLIP_DEFAULT,
        CATEGORY_AUGMENTATION_CONFIG_PATH,
        ROBBERY_CLASS,
        PREFLIGHT_MIN_ROBBERY_TRAIN_ROWS,
        PREFLIGHT_MIN_NEGATIVE_TRAIN_ROWS,
        PREFLIGHT_NEGATIVE_TO_ROBBERY_RATIO,
        PREFLIGHT_ROBBERY_DOMINANCE_THRESHOLD,
        PREFLIGHT_ROBBERY_RARE_THRESHOLD,
        DEFAULT_BINARY_SOFTMAX_THRESHOLD,
        DEFAULT_BINARY_LOGIT_MARGIN,
        TRAINING_PLAN_PATH,
    )
    from training_artifacts import resolve_artifacts, print_artifact_banner  # type: ignore[attr-defined]


# Semillas y splits
TRAIN_RATIO = SPLIT_RATIO_TRAIN
VAL_RATIO = SPLIT_RATIO_VAL  # resto será test
MIN_SEQ_LEN = 4   # descartar secuencias demasiado cortas


def _step_bounds(lo: float, hi: float, step: float) -> Tuple[float | None, float | None]:
    if step <= 0:
        return None, None
    first = math.ceil(lo / step) * step
    last = math.floor(hi / step) * step
    if first > last:
        return None, None
    return first, last


def _grid_values(lo: float, hi: float, step: float) -> List[float]:
    first, last = _step_bounds(lo, hi, step)
    if first is None or last is None:
        return []
    n = int(math.floor((last - first) / step + 1e-9)) + 1
    return [first + i * step for i in range(max(0, n))]


def build_deterministic_variant_specs(
    prof: Dict[str, Any],
    *,
    max_variants: int = 64,
    scale_lo: float = 95.0,
    scale_hi: float = 111.0,
    shift_abs: float = 0.06,
) -> List[Dict[str, Any]]:
    """
    Variantes deterministas alineadas con la rejilla por steps del perfil (como validate_npy).
    Cada spec: {"name": str, "ops": [ {"op": ...}, ... ] }
    """
    steps = prof.get("steps", {}) if isinstance(prof, dict) else {}
    rot_cfg = prof.get("rotate", {}) if isinstance(prof, dict) else {}
    noise_cfg = prof.get("noise", {}) if isinstance(prof, dict) else {}
    r_step = float(steps.get("rotate", 2.0))
    s_step = float(steps.get("scale", 2.0))
    sh_step = float(steps.get("shift", 0.02))
    r_lo = float(rot_cfg.get("min", -12.0))
    r_hi = float(rot_cfg.get("max", 12.0))
    sigma_cap = float(noise_cfg.get("sigma_cap", 0.006))

    specs: List[Dict[str, Any]] = [{"name": "identity", "ops": []}]
    specs.append({"name": "mirror", "ops": [{"op": "mirror"}]})

    speed_cfg = prof.get("speed", {}) if isinstance(prof, dict) else {}
    sp_lo = float(speed_cfg.get("min", AUGMENT_SPEED_FACTOR_LO))
    sp_hi = float(speed_cfg.get("max", AUGMENT_SPEED_FACTOR_HI))
    sp_step = float(steps.get("speed", 0.15))
    for fac in _grid_values(sp_lo, sp_hi, sp_step):
        if abs(fac - 1.0) < 1e-6:
            continue
        specs.append({"name": f"speed_{fac:.2f}", "ops": [{"op": "speed", "factor": float(fac)}]})

    for deg in _grid_values(r_lo, r_hi, r_step):
        if abs(deg) < 1e-9:
            continue
        specs.append({"name": f"rotate_{deg:.2f}", "ops": [{"op": "rotate", "deg": float(deg)}]})

    for pct in _grid_values(scale_lo, scale_hi, s_step):
        if abs(pct - 100.0) < 1e-6:
            continue
        specs.append({"name": f"scale_{pct:.2f}", "ops": [{"op": "scale", "pct": float(pct)}]})

    for dx in _grid_values(-shift_abs, shift_abs, sh_step):
        specs.append({"name": f"shift_x_{dx:.4f}", "ops": [{"op": "shift", "dx": float(dx), "dy": 0.0}]})
    for dy in _grid_values(-shift_abs, shift_abs, sh_step):
        specs.append({"name": f"shift_y_{dy:.4f}", "ops": [{"op": "shift", "dx": 0.0, "dy": float(dy)}]})

    specs.append(
        {
            "name": f"noise_{sigma_cap:.6f}",
            "ops": [{"op": "noise", "sx": sigma_cap, "sy": sigma_cap}],
        }
    )

    # Desduplicar por nombre y limitar tamaño (prioridad: identity, mirror, luego orden)
    seen = set()
    out: List[Dict[str, Any]] = []
    for s in specs:
        k = s["name"]
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
        if len(out) >= max(1, int(max_variants)):
            break
    return out


def _stable_uid_hash(uid: str) -> int:
    h = hashlib.md5(uid.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _stable_str_hash(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[8:16], 16)


def _apply_deterministic_ops(
    poses: np.ndarray,
    ops: List[Dict[str, Any]],
    rng: np.random.Generator,
) -> np.ndarray:
    out = poses.copy()
    for op in ops:
        kind = op.get("op")
        if kind == "mirror":
            out = _apply_mirror(out)
        elif kind == "rotate":
            out = _apply_rotate(out, float(op["deg"]))
        elif kind == "scale":
            out = _apply_scale(out, float(op["pct"]))
        elif kind == "shift":
            out = _apply_shift(out, float(op["dx"]), float(op["dy"]))
        elif kind == "noise":
            sx = float(op.get("sx", 0.0))
            sy = float(op.get("sy", 0.0))
            out = _apply_noise(out, rng, sx, sy)
        elif kind == "speed":
            out = _apply_speed(out, float(op["factor"]))
        np.clip(out, 0.0, 1.0, out=out)
    return out


# Caché de manifests por NPY (salida de validate_npy.py, un JSON por UID)
MANIFEST_CACHE_DIR = Path(__file__).parent / "operations_npy" / "manifest_cache"


def _flatten_validate_manifest_item(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convierte un ítem del manifest de validate_npy al formato interno de ops."""
    if not isinstance(item, dict):
        return []
    op = item.get("operation")
    params = item.get("params") or {}
    if op == "compose":
        pipeline = params.get("pipeline") or []
        out: List[Dict[str, Any]] = []
        for step in pipeline:
            out.extend(_flatten_validate_manifest_item(step))
        return out
    if op == "mirror":
        if params.get("apply", True):
            return [{"op": "mirror"}]
        return []
    if op == "rotate":
        return [{"op": "rotate", "deg": float(params["degrees"])}]
    if op == "scale":
        return [{"op": "scale", "pct": float(params["percentage"])}]
    if op == "shift":
        return [{"op": "shift", "dx": float(params["dx"]), "dy": float(params["dy"])}]
    if op == "noise":
        return [{"op": "noise", "sx": float(params["sigma_x"]), "sy": float(params["sigma_y"])}]
    if op == "speed":
        return [{"op": "speed", "factor": float(params["factor"])}]
    return []


def _variant_specs_from_validate_items(items: List[Dict[str, Any]], prefix: str) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for i, item in enumerate(items):
        ops = _flatten_validate_manifest_item(item)
        name = f"{prefix}_{i}_{item.get('operation', 'op')}"
        specs.append({"name": name, "ops": ops})
    return specs


def specs_from_validate_manifest_payload(
    data: Dict[str, Any],
    variant_set: str = MANIFEST_VARIANT_SET_DEFAULT,
) -> List[Dict[str, Any]]:
    """
    Construye lista de variantes deterministas desde un manifest.json de validate_npy.
    variant_set: 'min' | 'industrial' | 'full'
      - min: selected_n_min
      - industrial: selected_n_objetivo_industrial
      - full: industrial + mirror_composed + compose_light (deduplicado)
    Siempre incluye identity al inicio.
    """
    vs = str(variant_set).lower().strip()
    chunks: List[List[Dict[str, Any]]] = []
    if vs == "min":
        chunks.append(data.get("selected_n_min") or [])
    elif vs == "industrial":
        chunks.append(data.get("selected_n_objetivo_industrial") or [])
    elif vs == "full":
        chunks.append(data.get("selected_n_objetivo_industrial") or [])
        chunks.append(data.get("selected_n_objetivo_industrial_with_mirror_composed") or [])
        chunks.append(data.get("selected_n_objetivo_industrial_compose_light") or [])
    else:
        chunks.append(data.get("selected_n_objetivo_industrial") or [])

    seen_json = set()
    merged: List[Dict[str, Any]] = []
    for chunk in chunks:
        for it in chunk:
            key = json.dumps(it, sort_keys=True, ensure_ascii=False)
            if key in seen_json:
                continue
            seen_json.add(key)
            merged.append(it)

    out: List[Dict[str, Any]] = [{"name": "identity", "ops": []}]
    out.extend(_variant_specs_from_validate_items(merged, "m"))
    return out


def _output_base_data_result_from_experiments_config() -> Optional[Path]:
    """OUTPUT_BASE/data_result de experiments/config.py (misma salida que pose_extractor_clean)."""
    try:
        exp_dir = Path(__file__).resolve().parent.parent
        if str(exp_dir) not in sys.path:
            sys.path.insert(0, str(exp_dir))
        from config import OUTPUT_BASE  # type: ignore[import-untyped]

        if OUTPUT_BASE:
            return Path(OUTPUT_BASE).expanduser().resolve() / "data_result"
    except Exception:
        pass
    return None


def _candidate_data_result_roots(explicit: str | Path | None = None) -> List[Path]:
    candidates: List[Path] = []
    seen: set[str] = set()

    def _add(p: Path | None) -> None:
        if p is None:
            return
        rp = p.expanduser().resolve()
        key = str(rp)
        if key not in seen:
            seen.add(key)
            candidates.append(rp)

    if explicit is not None:
        _add(Path(explicit))
        return candidates

    env_root = os.environ.get("GUADIA_DATA_RESULT_ROOT", "").strip()
    if env_root:
        _add(Path(env_root))
    _add(Path(DATA_RESULT_ROOT))
    _add(_output_base_data_result_from_experiments_config())
    return candidates


def get_data_result_root(data_root: str | Path | None = None) -> Path:
    """
    Resuelve la carpeta data_result.
    Prioridad: argumento explícito → GUADIA_DATA_RESULT_ROOT → model_config → OUTPUT_BASE/data_result.
    """
    for root in _candidate_data_result_roots(data_root):
        if root.is_dir():
            return root
    tried = "\n    ".join(str(p) for p in _candidate_data_result_roots(data_root))
    raise RuntimeError(
        "No se encontró la carpeta data_result. Rutas probadas:\n"
        f"    {tried}\n"
        "Indica la ruta correcta con --data-root o exporta GUADIA_DATA_RESULT_ROOT."
    )


def _parse_category_dir_name(name: str) -> Optional[int]:
    s = str(name).strip()
    if not s:
        return None
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s)
    return None


def scan_data_result_folders(data_root: str | Path | None = None) -> Dict[int, Dict[str, int]]:
    """
    Inventario por nombre de carpeta bajo data_result/{cat}/ (sin filtros de calidad).
    Categoría = nombre de la carpeta si es entero (p. ej. 14).

    clip_dirs: subcarpetas bajo {cat}/ (aunque falte meta.json).
    clips: subcarpetas con meta.json (listas para train).
    """
    root = get_data_result_root(data_root)
    out: Dict[int, Dict[str, int]] = {}
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat = _parse_category_dir_name(cat_dir.name)
        if cat is None:
            continue
        clip_dirs = 0
        clips = 0
        users = 0
        for clip_dir in sorted(cat_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            clip_dirs += 1
            if not (clip_dir / "meta.json").is_file():
                continue
            clips += 1
            for ud in sorted(clip_dir.iterdir()):
                if not ud.is_dir() or not ud.name.startswith("user_"):
                    continue
                if (ud / "poses.npy").is_file() or (ud / "poses_full.npy").is_file():
                    users += 1
        out[cat] = {
            "clip_dirs": clip_dirs,
            "clips": clips,
            "users_with_poses": users,
        }
    return dict(sorted(out.items()))


def manifest_cache_path_for_uid(cache_dir: Path, uid: str) -> Path:
    h = hashlib.md5(uid.encode("utf-8")).hexdigest()
    return cache_dir / f"{h}.json"


def build_per_uid_variant_map(
    examples: List[PoseExample],
    manifest_cache_dir: Path,
    variant_set: str,
    verify_source_path: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Carga manifests desde manifest_cache_dir (un JSON por UID md5).
    Solo incluye UIDs con fichero válido; el resto usará fallback en PoseDataset.
    """
    manifest_cache_dir = Path(manifest_cache_dir)
    if not manifest_cache_dir.is_dir():
        return {}

    per_uid: Dict[str, List[Dict[str, Any]]] = {}
    for ex in examples:
        uid = _example_uid(ex)
        if uid in per_uid:
            continue
        p = resolve_manifest_json_for_example(manifest_cache_dir, ex)
        if p is None:
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if verify_source_path:
                src = data.get("source_npy")
                if src:
                    src_p = Path(str(src))
                    if not src_p.is_absolute():
                        src_p = get_data_result_root() / src_p
                    if src_p.resolve() != Path(ex.pose_path).resolve():
                        continue
            specs = specs_from_validate_manifest_payload(data, variant_set=variant_set)
            if specs:
                per_uid[uid] = specs
        except Exception:
            continue
    return per_uid


# Recetas de augmentación por categoría (orden de prioridad; se asignan de forma estable por clip).
CATEGORY_AUGMENT_RECIPE_BANK: List[Dict[str, Any]] = [
    {"name": "mirror", "ops": [{"op": "mirror"}]},
    {"name": "scale_small", "ops": [{"op": "scale", "pct": 92.0}]},
    {"name": "scale_large", "ops": [{"op": "scale", "pct": 108.0}]},
    {"name": "noise_light", "ops": [{"op": "noise", "sx": 0.003, "sy": 0.003}]},
    {"name": "speed_fast", "ops": [{"op": "speed", "factor": 0.88}]},
    {"name": "speed_slow", "ops": [{"op": "speed", "factor": 1.12}]},
    {"name": "rotate_left", "ops": [{"op": "rotate", "deg": -8.0}]},
    {"name": "rotate_right", "ops": [{"op": "rotate", "deg": 8.0}]},
    {"name": "shift_up", "ops": [{"op": "shift", "dx": 0.0, "dy": -0.04}]},
    {"name": "shift_down", "ops": [{"op": "shift", "dx": 0.0, "dy": 0.04}]},
    {"name": "mirror_scale_small", "ops": [{"op": "mirror"}, {"op": "scale", "pct": 94.0}]},
    {"name": "rotate_noise", "ops": [{"op": "rotate", "deg": 6.0}, {"op": "noise", "sx": 0.002, "sy": 0.002}]},
    {"name": "scale_speed_fast", "ops": [{"op": "scale", "pct": 96.0}, {"op": "speed", "factor": 0.9}]},
    {"name": "mirror_speed_slow", "ops": [{"op": "mirror"}, {"op": "speed", "factor": 1.1}]},
    {"name": "shift_noise", "ops": [{"op": "shift", "dx": -0.03, "dy": 0.02}, {"op": "noise", "sx": 0.0025, "sy": 0.0025}]},
]


def load_category_augmentation_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """
    Lee config_category_augmentation.json.
    categories: { "3": 5, ... } → N variantes augmentadas por clip de esa categoría.
    """
    path = Path(config_path or DEFAULT_CATEGORY_AUGMENTATION_CONFIG_PATH)
    if not path.is_file():
        return {"enabled": False, "default": 0, "categories": {}, "include_identity": True}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[CATEGORY-AUG] No se pudo leer {path}: {exc}")
        return {"enabled": False, "default": 0, "categories": {}, "include_identity": True}
    if not isinstance(data, dict):
        return {"enabled": False, "default": 0, "categories": {}, "include_identity": True}
    return data


def _category_augment_count_for_label(cfg: Dict[str, Any], action_label: int) -> int:
    cats = cfg.get("categories", {}) if isinstance(cfg.get("categories"), dict) else {}
    key = str(int(action_label))
    if key in cats and cats[key] is not None:
        try:
            return max(0, int(cats[key]))
        except (TypeError, ValueError):
            pass
    try:
        return max(0, int(cfg.get("default", 0)))
    except (TypeError, ValueError):
        return 0


def _category_augment_is_active(cfg: Dict[str, Any]) -> bool:
    if not cfg.get("enabled", True):
        return False
    cats = cfg.get("categories", {}) if isinstance(cfg.get("categories"), dict) else {}
    try:
        default_n = max(0, int(cfg.get("default", 0)))
    except (TypeError, ValueError):
        default_n = 0
    if default_n > 0:
        return True
    for v in cats.values():
        if v is None:
            continue
        try:
            if int(v) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _stable_permutation(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    idx = np.arange(size, dtype=np.int64)
    rng.shuffle(idx)
    return idx


def _random_ops_for_augment_slot(
    uid: str,
    slot: int,
    ranges: Dict[str, Tuple[float, float]],
) -> List[Dict[str, Any]]:
    """Genera 1–2 ops pseudo-aleatorias pero estables por clip+slot (si N supera el banco de recetas)."""
    seed = (_stable_uid_hash(uid) * 1009 + slot * 9176 + 31) & 0xFFFFFFFF
    rng = np.random.default_rng(int(seed))
    pool = ["mirror", "rotate", "scale", "shift", "noise", "speed"]
    n_ops = 1 if slot % 4 != 0 else 2
    chosen = rng.choice(pool, size=min(n_ops, len(pool)), replace=False)
    out: List[Dict[str, Any]] = []
    for op in chosen:
        if op == "mirror":
            out.append({"op": "mirror"})
        elif op == "rotate":
            lo, hi = ranges.get("rotate_degrees", (-12.0, 12.0))
            out.append({"op": "rotate", "deg": _rand_uniform(rng, lo, hi)})
        elif op == "scale":
            lo, hi = ranges.get("scale_percentage", (92.0, 110.0))
            out.append({"op": "scale", "pct": _rand_uniform(rng, lo, hi)})
        elif op == "shift":
            lox, hix = ranges.get("shift_dx", (-0.05, 0.05))
            loy, hiy = ranges.get("shift_dy", (-0.05, 0.05))
            out.append({"op": "shift", "dx": _rand_uniform(rng, lox, hix), "dy": _rand_uniform(rng, loy, hiy)})
        elif op == "noise":
            lox, hix = ranges.get("noise_sigma_x", (0.0, 0.004))
            loy, hiy = ranges.get("noise_sigma_y", (0.0, 0.004))
            out.append({"op": "noise", "sx": _rand_uniform(rng, lox, hix), "sy": _rand_uniform(rng, loy, hiy)})
        elif op == "speed":
            lo, hi = ranges.get("speed_factor", (AUGMENT_SPEED_FACTOR_LO, AUGMENT_SPEED_FACTOR_HI))
            out.append({"op": "speed", "factor": _rand_uniform(rng, lo, hi)})
    return out


def _category_augment_ops_for_clip(
    uid: str,
    n_variants: int,
    recipe_bank: List[Dict[str, Any]],
    ranges: Dict[str, Tuple[float, float]],
    seed: int = SEED,
) -> List[List[Dict[str, Any]]]:
    """Devuelve exactamente n_variants listas de ops (una por variante augmentada)."""
    if n_variants <= 0:
        return []
    bank = recipe_bank or CATEGORY_AUGMENT_RECIPE_BANK
    perm = _stable_permutation(len(bank), (_stable_uid_hash(uid) ^ int(seed)) & 0xFFFFFFFF)
    out: List[List[Dict[str, Any]]] = []
    for slot in range(n_variants):
        if slot < len(bank):
            recipe = bank[int(perm[slot % len(perm)])]
            out.append(list(recipe.get("ops", []) or []))
        else:
            out.append(_random_ops_for_augment_slot(uid, slot, ranges))
    return out


def expand_examples_with_category_augmentation(
    examples: List[PoseExample],
    cfg: Dict[str, Any],
    augment_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    recipe_bank: Optional[List[Dict[str, Any]]] = None,
    seed: int = SEED,
) -> List[PoseExample]:
    """
    Por cada clip base (forced_ops is None), genera variantes según config por categoría de acción.
    include_identity=true → fila original (forced_ops=[]) + N variantes con ops fijas.
    Solo debe invocarse sobre el split train (val/test sin augmentación).
    """
    if not _category_augment_is_active(cfg):
        return examples
    ranges = augment_ranges or {
        "rotate_degrees": (-12.0, 12.0),
        "scale_percentage": (92.0, 110.0),
        "shift_dx": (-0.05, 0.05),
        "shift_dy": (-0.05, 0.05),
        "noise_sigma_x": (0.0, 0.004),
        "noise_sigma_y": (0.0, 0.004),
        "speed_factor": (AUGMENT_SPEED_FACTOR_LO, AUGMENT_SPEED_FACTOR_HI),
    }
    include_identity = bool(cfg.get("include_identity", True))
    bank = recipe_bank or CATEGORY_AUGMENT_RECIPE_BANK

    out: List[PoseExample] = []
    expanded_clips = 0
    added_rows = 0
    by_cat: Dict[int, int] = {}

    for ex in examples:
        if ex.forced_ops is not None:
            out.append(ex)
            continue
        n_req = _category_augment_count_for_label(cfg, _example_folder_category(ex))
        if n_req <= 0:
            out.append(ex)
            continue

        uid = _example_uid(ex)
        ops_list = _category_augment_ops_for_clip(uid, n_req, bank, ranges, seed=seed)
        if include_identity:
            out.append(_copy_pose_example(ex, forced_ops=[]))
        else:
            out.append(ex)
        for ops in ops_list:
            out.append(_copy_pose_example(ex, forced_ops=list(ops)))
            added_rows += 1
        expanded_clips += 1
        cat = _example_folder_category(ex)
        by_cat[cat] = by_cat.get(cat, 0) + 1

    print(
        f"[CATEGORY-AUG] Clips expandidos: {expanded_clips} | "
        f"filas augmentadas añadidas: {added_rows} | "
        f"filas totales: {len(out)} (antes {len(examples)})"
    )
    if by_cat:
        summary = ", ".join(f"cat{k}={v}" for k, v in sorted(by_cat.items()))
        print(f"[CATEGORY-AUG] Clips expandidos por categoría de acción: {summary}")
    return out


def count_examples_by_action_label(examples: List[PoseExample]) -> Dict[int, int]:
    """Cuenta ejemplos por carpeta de categoría (data_result/{cat}/)."""
    return count_examples_by_folder_category(examples)


def count_examples_by_folder_category(examples: List[PoseExample]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for ex in examples:
        cat = _example_folder_category(ex)
        counts[cat] = counts.get(cat, 0) + 1
    return dict(sorted(counts.items()))


def train_rows_after_category_aug(n_clips: int, n_aug: int, include_identity: bool = True) -> int:
    """Filas en train tras expansión por categoría (sin contar manifest u otras)."""
    if n_clips <= 0:
        return 0
    if n_aug <= 0:
        return n_clips
    if include_identity:
        return n_clips * (1 + n_aug)
    return n_clips * n_aug


def _aug_for_target_rows(
    n_clips: int,
    target_rows: int,
    include_identity: bool,
    max_aug: int,
) -> int:
    """Variantes augmentadas necesarias para alcanzar ~target_rows filas en train."""
    n_clips = int(n_clips)
    target_rows = int(target_rows)
    max_aug = max(0, int(max_aug))
    if n_clips <= 0 or target_rows <= 0:
        return 0
    if target_rows <= n_clips:
        return 0
    if include_identity:
        needed = int(math.ceil(target_rows / n_clips)) - 1
    else:
        needed = int(math.ceil(target_rows / n_clips))
    return min(max_aug, max(0, needed))


def propose_category_augment_counts(
    train_counts: Dict[int, int],
    *,
    robbery_class: int = ROBBERY_CLASS,
    target_samples: Optional[int] = None,
    min_robbery_rows: Optional[int] = None,
    min_negative_rows: Optional[int] = None,
    negative_to_robbery_ratio: float = PREFLIGHT_NEGATIVE_TO_ROBBERY_RATIO,
    max_aug: int = 10,
    include_identity: bool = True,
) -> Dict[int, int]:
    """
    Propone augment por categoría con lógica asimétrica para detección de robo.

    - Clase robbery_class (p. ej. 6): priorizar recall (robos no perdidos) → más augment si hay pocos.
    - Resto de clases: priorizar reducir falsos positivos (acción normal → robo) → augment en minoritarias
      y refuerzo si el robo domina el train o el total de negativos queda bajo respecto al robo efectivo.

    Confundir 0↔1 u otras no-6 no se penaliza; solo importa no perder robos ni disparar FP hacia 6.
    """
    if not train_counts:
        return {}
    max_aug = max(0, int(max_aug))
    robbery_class = int(robbery_class)
    negative_to_robbery_ratio = max(1.0, float(negative_to_robbery_ratio))

    counts = {int(k): int(v) for k, v in train_counts.items()}
    n_rob = counts.get(robbery_class, 0)
    neg_counts = {k: v for k, v in counts.items() if k != robbery_class}
    neg_vals = [v for v in neg_counts.values() if v > 0]
    total_train = sum(counts.values())

    if min_negative_rows is None:
        min_negative_rows = PREFLIGHT_MIN_NEGATIVE_TRAIN_ROWS
    min_negative_rows = max(1, int(min_negative_rows))

    if target_samples is not None:
        base_target = max(1, int(target_samples))
    elif neg_vals:
        neg_median = sorted(neg_vals)[len(neg_vals) // 2]
        base_target = max(20, min(150, int(neg_median)))
    else:
        base_target = 40

    if min_robbery_rows is None:
        min_robbery_rows = max(PREFLIGHT_MIN_ROBBERY_TRAIN_ROWS, base_target)
    min_robbery_rows = max(1, int(min_robbery_rows))

    proposals: Dict[int, int] = {cat: 0 for cat in counts}

    # --- Fase A: recall de robo (evitar FN) ---
    if n_rob > 0:
        if n_rob >= min_robbery_rows * 2:
            proposals[robbery_class] = 0
        else:
            proposals[robbery_class] = _aug_for_target_rows(
                n_rob, min_robbery_rows, include_identity, max_aug
            )
        rob_share = n_rob / max(1, total_train)
        if rob_share < PREFLIGHT_ROBBERY_RARE_THRESHOLD and proposals[robbery_class] < max_aug:
            proposals[robbery_class] = min(max_aug, proposals[robbery_class] + 1)
    else:
        proposals[robbery_class] = 0

    effective_rob = train_rows_after_category_aug(
        n_rob, proposals.get(robbery_class, 0), include_identity
    )
    target_total_neg = max(
        len(neg_counts) * min_negative_rows,
        int(effective_rob * negative_to_robbery_ratio),
    )

    # --- Fase B: negativos (evitar FP hacia robo) ---
    if neg_counts:
        per_class_floor = max(min_negative_rows, target_total_neg // max(1, len(neg_counts)))
        for cat, n in neg_counts.items():
            if n <= 0:
                continue
            target_neg = per_class_floor
            if n >= target_neg:
                proposals[cat] = 0
            else:
                proposals[cat] = _aug_for_target_rows(n, target_neg, include_identity, max_aug)

    # --- Fase C: robo muy frecuente en bruto → más diversidad negativa (FP) ---
    if n_rob > 0 and total_train > 0:
        rob_share = n_rob / total_train
        if rob_share > PREFLIGHT_ROBBERY_DOMINANCE_THRESHOLD:
            for cat in neg_counts:
                if proposals.get(cat, 0) < max_aug:
                    proposals[cat] = min(max_aug, proposals.get(cat, 0) + 1)

    return proposals


def analyze_robbery_augment_balance(
    train_counts: Dict[int, int],
    proposals: Dict[int, int],
    *,
    robbery_class: int = ROBBERY_CLASS,
    include_identity: bool = True,
    negative_to_robbery_ratio: float = PREFLIGHT_NEGATIVE_TO_ROBBERY_RATIO,
) -> Dict[str, Any]:
    """Métricas de balance robo vs negativos tras una propuesta de augment."""
    robbery_class = int(robbery_class)
    counts = {int(k): int(v) for k, v in train_counts.items()}
    props = {int(k): int(v) for k, v in proposals.items()}
    n_rob = counts.get(robbery_class, 0)
    rob_rows = train_rows_after_category_aug(
        n_rob, props.get(robbery_class, 0), include_identity
    )
    neg_rows = 0
    for cat, n in counts.items():
        if cat == robbery_class:
            continue
        neg_rows += train_rows_after_category_aug(n, props.get(cat, 0), include_identity)
    ratio = neg_rows / max(1, rob_rows)
    total = sum(counts.values())
    return {
        "robbery_class": robbery_class,
        "robbery_clips_train": n_rob,
        "robbery_rows_effective": rob_rows,
        "negative_rows_effective": neg_rows,
        "negative_to_robbery_ratio": float(ratio),
        "negative_to_robbery_target": float(negative_to_robbery_ratio),
        "robbery_share_raw": (n_rob / total) if total else 0.0,
        "warnings": [
            *(["Sin clips de robo en train: imposible entrenar detección de clase 6."] if n_rob == 0 else []),
            *(
                ["Robo muy raro en train: revisa augment de clase 6 y métricas de recall."]
                if total and (n_rob / total) < 0.05
                else []
            ),
            *(
                [f"Ratio neg/robo efectivo ({ratio:.1f}) por debajo del objetivo ({negative_to_robbery_ratio:.1f}): riesgo de FP."]
                if n_rob > 0 and ratio < negative_to_robbery_ratio * 0.85
                else []
            ),
            *(
                ["Robo domina el train en bruto (>25%): se ha reforzado augment de negativos."]
                if total and (n_rob / total) > 0.25
                else []
            ),
        ],
    }


def summarize_category_aug_on_train(
    train_counts: Dict[int, int],
    cfg: Dict[str, Any],
) -> Dict[int, Dict[str, int]]:
    """Por categoría: clips train originales y filas tras expansión según cfg."""
    include_identity = bool(cfg.get("include_identity", True))
    out: Dict[int, Dict[str, int]] = {}
    for cat, n in train_counts.items():
        n_aug = _category_augment_count_for_label(cfg, int(cat))
        out[int(cat)] = {
            "clips": int(n),
            "aug_per_clip": int(n_aug),
            "train_rows": train_rows_after_category_aug(int(n), int(n_aug), include_identity),
        }
    return out


def expand_examples_with_manifest_extra_views(
    examples: List[PoseExample],
    mdir: Path,
    variant_set: str,
    extra_views_per_clip: int,
) -> List[PoseExample]:
    """
    Por cada clip con manifest validate_npy en caché, genera 1 + hasta `extra_views_per_clip`
    entradas de dataset (misma etiqueta, mismo .npy): identidad (ops []) + las primeras variantes
    del manifest. No escribe ficheros; solo fija forced_ops para __getitem__.
    Clips sin manifest se dejan en un solo ejemplo (comportamiento anterior).
    """
    if extra_views_per_clip <= 0:
        return examples
    mdir = Path(mdir)
    if not mdir.is_dir():
        print("[EXPAND-VIEWS] manifest_cache_dir no es una carpeta válida; no se expande.")
        return examples
    n_req = int(extra_views_per_clip)
    out: List[PoseExample] = []
    skipped_no_json = 0
    n_with_json = 0
    n_ge_n = 0  # len(specs)-1 >= n_req (pueden generarse N variantes extra)
    n_partial = 0  # 0 < disponibles < n_req
    n_only_identity = 0  # specs solo identidad o una fila
    for ex in examples:
        if ex.forced_ops is not None:
            out.append(ex)
            continue
        p = resolve_manifest_json_for_example(mdir, ex)
        if p is None:
            out.append(ex)
            skipped_no_json += 1
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            out.append(ex)
            skipped_no_json += 1
            continue
        specs = specs_from_validate_manifest_payload(data, variant_set=variant_set)
        if not specs:
            out.append(ex)
            skipped_no_json += 1
            continue
        n_with_json += 1
        avail = max(0, len(specs) - 1)
        if avail >= n_req:
            n_ge_n += 1
        elif avail > 0:
            n_partial += 1
        else:
            n_only_identity += 1
        out.append(_copy_pose_example(ex, forced_ops=[]))
        n_extra = min(n_req, avail)
        for j in range(n_extra):
            spec = specs[1 + j]
            ops = list(spec.get("ops", []) or [])
            out.append(_copy_pose_example(ex, forced_ops=ops))
    print(
        f"[EXPAND-VIEWS] Solicitadas {n_req} variantes extra por clip (además de identidad) | "
        f"variant_set={variant_set}"
    )
    print(
        f"[EXPAND-VIEWS] Cobertura: con JSON válido={n_with_json} | "
        f"con>={n_req} variantes extra en manifest={n_ge_n} | "
        f"parcial (algunas pero <{n_req})={n_partial} | "
        f"solo identidad en specs={n_only_identity} | sin JSON={skipped_no_json}"
    )
    if n_req > 0 and n_with_json > 0 and n_ge_n < n_with_json:
        print(
            f"[EXPAND-VIEWS] Aviso: solo {n_ge_n}/{n_with_json} clips tienen al menos {n_req} variantes extra "
            "en el JSON; el resto usa menos filas. Regenera caché con batch_build_manifest_cache.py "
            "o baja --extra-manifest-views-per-clip."
        )
    print(
        f"[EXPAND-VIEWS] Filas dataset: {len(out)} (antes {len(examples)}) "
        f"| identidad + hasta {n_req} ops forzadas por clip con manifest suficiente"
    )
    return out


def build_pose_dataset_for_eval(
    examples: List[PoseExample],
    label_to_idx: Dict[int, int],
    seq_len: int,
    dataset_split: str,
    checkpoint: Dict[str, Any],
) -> "PoseDataset":
    """
    Dataset de evaluación alineado con el checkpoint (misma rejilla o caché de manifests).
    """
    augment_profile = checkpoint.get("augment_profile", AUGMENT_PROFILE_DEFAULT)
    aug_path = Path(checkpoint.get("augment_config_path", str(AUGMENT_CONFIG_PATH)))
    prof = _load_augment_profile(aug_path, str(augment_profile))
    max_v = int(checkpoint.get("deterministic_variants_count") or MAX_DETERMINISTIC_VARIANTS)
    det_specs = build_deterministic_variant_specs(prof if prof else {}, max_variants=max_v)
    mdir = checkpoint.get("manifest_cache_dir")
    variant_set = str(checkpoint.get("manifest_variant_set", MANIFEST_VARIANT_SET_DEFAULT))
    per_uid: Optional[Dict[str, List[Dict[str, Any]]]] = None
    if mdir:
        per_uid = build_per_uid_variant_map(
            examples,
            Path(str(mdir)),
            variant_set=variant_set,
            verify_source_path=True,
        )
        if not per_uid:
            per_uid = None
    return PoseDataset(
        examples,
        label_to_idx,
        seq_len,
        augment_on_the_fly=False,
        dataset_split=dataset_split,
        deterministic_variants=det_specs,
        per_uid_variants=per_uid,
        augment_seed=SEED,
    )


# MIN_CLIP_SECONDS, MIN_VALID_FRAMES, MIN_VALID_PCT, MAX_OCCLUSION_RATIO: model_config

# Modo debug: usar muy pocos datos y un experimento por arquitectura
DEBUG_MODE = False          # ponlo a True en local para pruebas rápidas
DEBUG_MAX_EXAMPLES = 5      # cuántos embeddings usar en total en debug

# Directorios locales para modelos y logs (dentro de training/)
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models-operation"
MODELS_SINGLE_DIR = BASE_DIR / "models-operation-single"
LOGS_DIR = BASE_DIR / "logs" 
SPLITS_DIR = BASE_DIR / "splits"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_SINGLE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# Augment on-the-fly (sin crear ficheros .npy en disco); AUGMENT_PROFILE_DEFAULT en model_config
AUGMENT_CONFIG_PATH = BASE_DIR / "operations_npy" / "validate_npy.json"
DEFAULT_CATEGORY_AUGMENTATION_CONFIG_PATH = CATEGORY_AUGMENTATION_CONFIG_PATH

# Un solo bloque de resumen de pool al primer build_datasets_and_loaders de la sesión
_POOL_SUMMARY_PRINTED = False


def reset_training_pool_summary_flag() -> None:
    global _POOL_SUMMARY_PRINTED
    _POOL_SUMMARY_PRINTED = False


def _summarize_forced_rows(examples: List[PoseExample]) -> Tuple[int, int, int]:
    """Devuelve (solo_rejilla, manifest_identidad, manifest_con_ops)."""
    grid_only = 0
    man_id = 0
    man_ops = 0
    for ex in examples:
        fo = getattr(ex, "forced_ops", None)
        if fo is None:
            grid_only += 1
        elif len(fo) == 0:
            man_id += 1
        else:
            man_ops += 1
    return grid_only, man_id, man_ops


def _print_training_pool_summary_once(
    *,
    train_unique_npy: int,
    train_rows: int,
    train_rows_before_expand: int,
    extra_manifest_views_per_clip: int,
    val_unique_npy: int,
    val_rows: int,
    test_unique_npy: int,
    test_rows: int,
    n_det_grid_specs: int,
    manifest_hits: int,
    train_grid_only: int,
    train_manifest_identity: int,
    train_manifest_ops: int,
    augment_on_the_fly: bool,
    augment_prob: float,
    use_deterministic_in_train: bool,
    train_deterministic_prob: float,
) -> None:
    global _POOL_SUMMARY_PRINTED
    if _POOL_SUMMARY_PRINTED:
        return
    _POOL_SUMMARY_PRINTED = True
    tr_g, tr_mi, tr_mo = train_grid_only, train_manifest_identity, train_manifest_ops
    det_rows = tr_mi + tr_mo
    print("\n" + "=" * 80)
    print("[RESUMEN POOL DE ENTRENAMIENTO] (primer experimento de esta sesión)")
    print("=" * 80)
    print(
        f"  · NPY únicos en split train (ficheros .npy distintos): {train_unique_npy}\n"
        f"  · Filas dataset TRAIN: {train_rows} "
        f"(antes de expansión manifest: {train_rows_before_expand}"
        + (
            f"; +{train_rows - train_rows_before_expand} filas por --extra-manifest-views-per-clip={extra_manifest_views_per_clip})"
            if extra_manifest_views_per_clip > 0
            else ")"
        )
    )
    print(
        f"  · Desglose filas TRAIN: rejilla global (__getitem__, sin forced_ops)={tr_g} | "
        f"manifest explícito identidad={tr_mi} | manifest explícito con ops={tr_mo} "
        f"(deterministas de dataset por fila: {det_rows})"
    )
    print(
        f"  · Val: NPY únicos={val_unique_npy} | filas loader={val_rows} | "
        f"Test: NPY únicos={test_unique_npy} | filas loader={test_rows}"
    )
    print(
        f"  · Rejilla determinista validate_npy (variantes en perfil): {n_det_grid_specs} | "
        f"UIDs con JSON en manifest_cache: {manifest_hits}"
    )
    print(
        "  · En cada época TRAIN: "
        + (
            f"augment ALEATORIO on-the-fly activo (prob≈{augment_prob} por pasada según config)"
            if augment_on_the_fly
            else "augment aleatorio on-the-fly DESACTIVADO"
        )
        + " | "
        + (
            f"variantes deterministas de rejilla en train con prob {train_deterministic_prob}"
            if use_deterministic_in_train and n_det_grid_specs > 0
            else ("sin variantes deterministas en train" if not use_deterministic_in_train else "sin rejilla")
        )
    )
    print(
        "  · Nota: las filas 'manifest explícito' aplican ops fijas por fila; "
        "la rejilla elige una variante determinista por muestra; el aleatorio es adicional si on-the-fly."
    )
    print("=" * 80 + "\n")


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class PoseExample:
    pose_path: Path
    label: int          # clase (original o binaria, según el modo)
    track_id: int
    clip_name: str
    category_str: str   # por si quieres inspeccionar
    valid_mask_path: Optional[Path] = None  # si usa poses_full.npy: máscara de frames válidos (sin NaN)
    users_in_clip: int = 1  # número de usuarios en meta["users"] para este clip
    action_label: Optional[int] = None  # categoría de acción original (persiste en modo binario)
    # Si no es None: aplicar solo estas ops deterministas (manifest validate_npy); train puede añadir on-the-fly después.
    forced_ops: Optional[List[Dict[str, Any]]] = None


def _example_action_label(ex: PoseExample) -> int:
    if ex.action_label is not None:
        return int(ex.action_label)
    return int(ex.label)


def _example_folder_category(ex: PoseExample) -> int:
    """Categoría según carpeta data_result/{cat}/ (category_str), no label de entrenamiento."""
    cat = _parse_category_dir_name(ex.category_str)
    if cat is not None:
        return cat
    return _example_action_label(ex)


def _copy_pose_example(
    ex: PoseExample,
    forced_ops: Optional[List[Dict[str, Any]]],
) -> PoseExample:
    return PoseExample(
        pose_path=ex.pose_path,
        label=ex.label,
        track_id=ex.track_id,
        clip_name=ex.clip_name,
        category_str=ex.category_str,
        valid_mask_path=ex.valid_mask_path,
        users_in_clip=ex.users_in_clip,
        action_label=_example_action_label(ex),
        forced_ops=forced_ops,
    )


def _manifest_lookup_uids(ex: PoseExample) -> List[str]:
    """
    Candidatos de UID para manifest_cache (md5 por string UTF-8).
    1) Ruta relativa posix bajo DATA_RESULT_ROOT (portable entre máquinas).
    2) Ruta absoluta resuelta (compatibilidad con cachés y splits antiguos).
    """
    root = get_data_result_root().resolve()
    pp = Path(ex.pose_path).resolve()
    out: List[str] = []
    try:
        out.append(pp.relative_to(root).as_posix())
    except ValueError:
        pass
    abs_s = str(pp)
    if not out or out[-1] != abs_s:
        out.append(abs_s)
    seen: set[str] = set()
    uniq: List[str] = []
    for u in out:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def _example_uid(ex: PoseExample) -> str:
    # UID estable para split/eval: relativo a data_result si aplica; si no, absoluto.
    uids = _manifest_lookup_uids(ex)
    return uids[0]


def example_in_split_set(ex: PoseExample, split_uids: set[str]) -> bool:
    """True si el ejemplo coincide con un UID del split (relativo nuevo o absoluto legado)."""
    for u in _manifest_lookup_uids(ex):
        if u in split_uids:
            return True
    return False


def resolve_manifest_json_for_example(cache_dir: Path, ex: PoseExample) -> Optional[Path]:
    """Primer manifest JSON encontrado para este .npy (relativo o legado)."""
    mdir = Path(cache_dir)
    if not mdir.is_dir():
        return None
    for uid in _manifest_lookup_uids(ex):
        p = manifest_cache_path_for_uid(mdir, uid)
        if p.is_file():
            return p
    return None


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _user_quality_ok(
    user_meta: Dict[str, Any],
    meta: Dict[str, Any],
    pose_len: int,
    min_clip_seconds: float = MIN_CLIP_SECONDS,
    min_valid_frames: int = MIN_VALID_FRAMES,
    min_valid_pct: float = MIN_VALID_PCT,
    max_occlusion_ratio: float = MAX_OCCLUSION_RATIO,
) -> bool:
    """
    Filtro de calidad para evitar meter ruido al modelo.
    """
    valid_frames = _to_int(user_meta.get("valid_frames"), default=pose_len)
    total_frames = _to_int(user_meta.get("total_frames"), default=pose_len)
    valid_pct = _to_float(user_meta.get("valid_pct"), default=100.0 if total_frames <= 0 else 0.0)
    occlusion_ratio = _to_float(user_meta.get("occlusion_ratio"), default=0.0)

    if total_frames <= 0:
        total_frames = pose_len

    # Si no viene valid_pct pero sí frames, lo inferimos.
    if valid_pct <= 0 and total_frames > 0 and valid_frames > 0:
        valid_pct = 100.0 * (valid_frames / total_frames)

    clip_duration = _to_float(meta.get("clip_duration"), default=0.0)
    if min_clip_seconds > 0 and clip_duration > 0 and clip_duration < min_clip_seconds:
        return False

    if pose_len < MIN_SEQ_LEN:
        return False
    if min_valid_frames > 0 and valid_frames < min_valid_frames:
        return False
    if min_valid_pct > 0 and valid_pct < min_valid_pct:
        return False
    if occlusion_ratio > max_occlusion_ratio:
        return False

    # No usamos meta["passes_filters"] del extractor: el entrenamiento se gobierna solo con los
    # umbrales numéricos (model_config MIN_* / MAX_OCCLUSION_* y CLI), alineados con preflight.

    return True


def _load_augment_profile(config_path: Path, profile: str) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return {}
    profiles = cfg.get("profiles", {})
    prof = profiles.get(profile, {})
    return prof if isinstance(prof, dict) else {}


def _rand_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    if hi < lo:
        lo, hi = hi, lo
    return float(rng.uniform(lo, hi))


def _apply_mirror(poses: np.ndarray) -> np.ndarray:
    out = poses.copy()
    out[..., 0] = 1.0 - out[..., 0]
    # KEEP_KPS = [5,6,7,8,9,10,11,12]:
    # [L-shoulder, R-shoulder, L-elbow, R-elbow, L-wrist, R-wrist, L-hip, R-hip]
    # Al espejar horizontalmente, hay que intercambiar pares izquierda/derecha.
    if out.ndim >= 3 and out.shape[-2] == 8:
        lr_pairs = ((0, 1), (2, 3), (4, 5), (6, 7))
        for li, ri in lr_pairs:
            tmp = out[..., li, :].copy()
            out[..., li, :] = out[..., ri, :]
            out[..., ri, :] = tmp
    return out


def _apply_rotate(poses: np.ndarray, degrees: float) -> np.ndarray:
    out = poses.copy()
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    x = out[..., 0] - 0.5
    y = out[..., 1] - 0.5
    xr = x * c - y * s
    yr = x * s + y * c
    out[..., 0] = xr + 0.5
    out[..., 1] = yr + 0.5
    return out


def _apply_scale(poses: np.ndarray, percentage: float) -> np.ndarray:
    factor = percentage / 100.0
    out = poses.copy()
    out[..., 0] = (out[..., 0] - 0.5) * factor + 0.5
    out[..., 1] = (out[..., 1] - 0.5) * factor + 0.5
    return out


def _apply_shift(poses: np.ndarray, dx: float, dy: float) -> np.ndarray:
    out = poses.copy()
    out[..., 0] = out[..., 0] + dx
    out[..., 1] = out[..., 1] + dy
    return out


def _apply_noise(poses: np.ndarray, rng: np.random.Generator, sigma_x: float, sigma_y: float) -> np.ndarray:
    out = poses.copy()
    out[..., 0] = out[..., 0] + rng.normal(loc=0.0, scale=max(0.0, sigma_x), size=out[..., 0].shape)
    out[..., 1] = out[..., 1] + rng.normal(loc=0.0, scale=max(0.0, sigma_y), size=out[..., 1].shape)
    return out


def _apply_speed(poses: np.ndarray, factor: float) -> np.ndarray:
    """
    Perturbación temporal sobre [T, J, D]: interpola la secuencia estirando/comprimiendo el eje T.
    factor > 1 → acción más lenta (más frames); factor < 1 → más rápida (menos frames).
    """
    if factor <= 0 or abs(factor - 1.0) < 1e-6:
        return poses
    t = poses.shape[0]
    if t < 2:
        return poses
    new_t = max(2, int(round(t * factor)))
    if new_t == t:
        return poses
    old_idx = np.arange(t, dtype=np.float64)
    new_idx = np.linspace(0.0, t - 1.0, new_t)
    flat = poses.reshape(t, -1)
    out_flat = np.empty((new_t, flat.shape[1]), dtype=poses.dtype)
    for col in range(flat.shape[1]):
        out_flat[:, col] = np.interp(new_idx, old_idx, flat[:, col])
    return out_flat.reshape(new_t, *poses.shape[1:])


def _augment_poses_on_the_fly(
    poses: np.ndarray,
    rng: np.random.Generator,
    augment_prob: float,
    max_ops: int,
    op_probs: Dict[str, float],
    ranges: Dict[str, Tuple[float, float]],
) -> np.ndarray:
    if augment_prob <= 0 or rng.random() >= augment_prob:
        return poses
    ops = ["mirror", "rotate", "scale", "shift", "noise", "speed"]
    probs = np.array([max(0.0, float(op_probs.get(op, 0.0))) for op in ops], dtype=np.float64)
    if probs.sum() <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / probs.sum()
    n_ops = int(rng.integers(1, max(2, max_ops + 1)))
    chosen = rng.choice(ops, size=n_ops, replace=False if n_ops <= len(ops) else True, p=probs)
    out = poses.copy()
    for op in chosen:
        if op == "mirror":
            out = _apply_mirror(out)
        elif op == "rotate":
            lo, hi = ranges.get("rotate_degrees", (-10.0, 10.0))
            out = _apply_rotate(out, _rand_uniform(rng, lo, hi))
        elif op == "scale":
            lo, hi = ranges.get("scale_percentage", (95.0, 105.0))
            out = _apply_scale(out, _rand_uniform(rng, lo, hi))
        elif op == "shift":
            lox, hix = ranges.get("shift_dx", (-0.02, 0.02))
            loy, hiy = ranges.get("shift_dy", (-0.02, 0.02))
            out = _apply_shift(out, _rand_uniform(rng, lox, hix), _rand_uniform(rng, loy, hiy))
        elif op == "noise":
            lox, hix = ranges.get("noise_sigma_x", (0.0, 0.003))
            loy, hiy = ranges.get("noise_sigma_y", (0.0, 0.003))
            out = _apply_noise(out, rng, _rand_uniform(rng, lox, hix), _rand_uniform(rng, loy, hiy))
        elif op == "speed":
            lo, hi = ranges.get(
                "speed_factor",
                (AUGMENT_SPEED_FACTOR_LO, AUGMENT_SPEED_FACTOR_HI),
            )
            out = _apply_speed(out, _rand_uniform(rng, lo, hi))
    np.clip(out, 0.0, 1.0, out=out)
    return out


def collect_examples(
    pose_source: str = "filtered",
    single_user_only: bool = False,
    min_clip_seconds: float = MIN_CLIP_SECONDS,
    min_valid_frames: int = MIN_VALID_FRAMES,
    min_valid_pct: float = MIN_VALID_PCT,
    max_occlusion_ratio: float = MAX_OCCLUSION_RATIO,
    data_root: str | Path | None = None,
) -> List[PoseExample]:
    """
    Recorre data_result/{cat}/{clip_name}/ y construye ejemplos por usuario:
      1) Incluye usuarios tanto en clips de 1 persona como multiusuario (por defecto).
      2) Si single_user_only=True: solo clips con exactamente 1 usuario.
      3) En cat=6 respeta user_cat por usuario:
         - user_cat=6 se mantiene como robo
         - user_cat!=6 se reetiqueta a su user_cat.
      4) En no-cat6 también prioriza user_cat si existe, para etiqueta por usuario.
      5) Aplica filtro de calidad por usuario (_user_quality_ok: umbrales model_config / CLI).
      6) pose_source: "filtered" usa poses.npy, "full" usa poses_full.npy (+ valid_mask).
      preflight_check_operations.py usa esta misma función para N y tiempos estimados.
    """
    root = get_data_result_root(data_root)
    examples: List[PoseExample] = []

    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat_str = cat_dir.name
        folder_cat = _parse_category_dir_name(cat_str)
        if folder_cat is None:
            continue
        for clip_dir in sorted(cat_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            meta_path = clip_dir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue

            users = meta.get("users", [])
            if not users:
                continue

            # Filtrado opcional para modo "solo usuario único".
            if single_user_only:
                if len(users) != 1:
                    continue
            clip_cat_int = _to_int(meta.get("cat", cat_str), default=_to_int(cat_str, default=0))
            # Regla de selección por clip:
            # - cat != 6: usar todos los usuarios válidos.
            # - cat == 6: también usar todos los usuarios válidos.
            #   La etiqueta final se resuelve por user_cat (robos=6, resto a su clase no-6).
            users_selected = users

            for user in users_selected:
                track_id = user.get("track_id")
                if track_id is None:
                    continue

                user_dir = clip_dir / f"user_{track_id}"
                pose_filename = "poses.npy" if pose_source == "filtered" else "poses_full.npy"
                pose_path = user_dir / pose_filename
                if not pose_path.exists():
                    continue

                try:
                    poses = np.load(pose_path)
                except Exception:
                    continue

                if poses.ndim != 3 or poses.shape[-1] != 2:
                    continue

                valid_mask_path = None
                effective_len = poses.shape[0]
                if pose_source == "full":
                    mask_path = user_dir / "valid_mask.npy"
                    if mask_path.exists():
                        try:
                            valid_mask = np.load(mask_path)
                            if valid_mask.ndim != 1 or len(valid_mask) != poses.shape[0]:
                                continue
                            effective_len = int(valid_mask.sum())
                            if effective_len < MIN_SEQ_LEN:
                                continue
                            valid_mask_path = mask_path
                        except Exception:
                            continue
                    elif np.any(np.isnan(poses)):
                        continue

                if not _user_quality_ok(
                    user,
                    meta,
                    effective_len,
                    min_clip_seconds=min_clip_seconds,
                    min_valid_frames=min_valid_frames,
                    min_valid_pct=min_valid_pct,
                    max_occlusion_ratio=max_occlusion_ratio,
                ):
                    continue

                # Etiqueta final por usuario:
                # - prioriza user_cat (si está)
                # - fallback a cat global del clip/carpeta
                user_cat = user.get("user_cat")
                if user_cat is not None:
                    label = _to_int(user_cat, default=_to_int(meta.get("cat", cat_str), default=0))
                else:
                    label = _to_int(meta.get("cat", cat_str), default=0)

                examples.append(
                    PoseExample(
                        pose_path=pose_path,
                        label=label,
                        track_id=int(track_id),
                        clip_name=str(meta.get("clip_name", clip_dir.name)),
                        category_str=cat_str,
                        valid_mask_path=valid_mask_path,
                        users_in_clip=int(len(users)),
                        action_label=int(folder_cat),
                    )
                )

    if not examples:
        raise RuntimeError("No se encontraron ejemplos válidos en data_result.")
    return examples


def normalize_sequence(poses: np.ndarray) -> np.ndarray:
    """
    poses: [T, J, 2] con coordenadas normalizadas 0-1.
    Centra por la media de joints y escala por tamaño medio del cuerpo.
    """
    poses = poses.astype(np.float32)
    center = poses.mean(axis=1, keepdims=True)
    poses = poses - center
    scale = np.linalg.norm(poses, axis=-1).mean()
    if scale > 0:
        poses = poses / scale
    return poses


def add_velocity(poses: np.ndarray) -> np.ndarray:
    """
    poses: [T, J, 2] -> concatena velocidad: [T, J, 4] con (x,y,dx,dy).
    """
    vel = np.diff(poses, axis=0, prepend=poses[0:1])
    return np.concatenate([poses, vel], axis=-1)


def temporal_resize(seq: np.ndarray, target_len: int) -> np.ndarray:
    """
    Redimensiona temporalmente una secuencia [T, ...] a [target_len, ...]
    con muestreo uniforme o padding por repetición.
    """
    t = seq.shape[0]
    if t == target_len:
        return seq
    if t > target_len:
        idx = np.linspace(0, t - 1, target_len).round().astype(int)
        return seq[idx]
    # t < target_len: padding repitiendo último frame
    pad_len = target_len - t
    pad = np.repeat(seq[-1:], pad_len, axis=0)
    return np.concatenate([seq, pad], axis=0)


class PoseDataset(Dataset):
    """
    Dataset que aplica:
      - normalización espacial
      - augment espacial/temporal (mirror, rotate, scale, shift, noise, speed)
      - concatenación de velocidades
      - resize temporal a seq_len
      - flatten de joints a un vector de features por frame

    Política de augment:
      - split="train": opcional (1) variantes deterministas (rejilla validate_npy) y (2) augment aleatorio
      - split="val"/"test": solo variantes deterministas (sin muestreo aleatorio en rangos)
    """

    def __init__(
        self,
        examples: List[PoseExample],
        label_to_idx: Dict[int, int],
        seq_len: int,
        augment_on_the_fly: bool = False,
        augment_prob: float = AUGMENT_PROB,
        augment_max_ops: int = AUGMENT_MAX_OPS,
        augment_op_probs: Optional[Dict[str, float]] = None,
        augment_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        augment_seed: int = SEED,
        dataset_split: str = "train",
        deterministic_variants: Optional[List[Dict[str, Any]]] = None,
        train_deterministic_prob: float = TRAIN_DETERMINISTIC_PROB,
        use_deterministic_in_train: bool = True,
        per_uid_variants: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ):
        self.examples = examples
        self.label_to_idx = label_to_idx
        self.seq_len = seq_len
        self.augment_on_the_fly = augment_on_the_fly
        self.augment_prob = augment_prob
        self.augment_max_ops = max(1, augment_max_ops)
        self.augment_op_probs = augment_op_probs or {
            "mirror": 0.26,
            "rotate": 0.24,
            "scale": 0.14,
            "shift": 0.14,
            "noise": 0.08,
            "speed": 0.14,
        }
        self.augment_ranges = augment_ranges or {
            "rotate_degrees": (-15.0, 15.0),
            "scale_percentage": (95.0, 110.0),
            "shift_dx": (-0.05, 0.05),
            "shift_dy": (-0.05, 0.05),
            "noise_sigma_x": (0.0, 0.006),
            "noise_sigma_y": (0.0, 0.006),
            "speed_factor": (AUGMENT_SPEED_FACTOR_LO, AUGMENT_SPEED_FACTOR_HI),
        }
        self.augment_seed = int(augment_seed)
        self.dataset_split = dataset_split
        self.deterministic_variants = deterministic_variants or [{"name": "identity", "ops": []}]
        self.train_deterministic_prob = float(max(0.0, min(1.0, train_deterministic_prob)))
        self.use_deterministic_in_train = bool(use_deterministic_in_train)
        self.per_uid_variants = per_uid_variants or {}

    def _variants_for_uid(self, uid: str) -> List[Dict[str, Any]]:
        if self.per_uid_variants and uid in self.per_uid_variants:
            return self.per_uid_variants[uid]
        return self.deterministic_variants

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        ex = self.examples[idx]
        poses = np.load(ex.pose_path)  # [T, J, 2]
        if ex.valid_mask_path is not None and ex.valid_mask_path.exists():
            valid_mask = np.load(ex.valid_mask_path)
            poses = poses[valid_mask].copy()
        if np.any(np.isnan(poses)):
            poses = np.nan_to_num(poses, nan=0.0, posinf=0.0, neginf=0.0)

        uid = _example_uid(ex)
        forced = getattr(ex, "forced_ops", None)
        if forced is not None:
            det_seed = (
                _stable_uid_hash(uid) * 1009 + _stable_str_hash(repr(forced)) + int(idx) * 131
            ) & 0xFFFFFFFF
            rng_det = np.random.default_rng(int(det_seed))
            poses = _apply_deterministic_ops(poses, forced, rng=rng_det)
            if self.dataset_split == "train" and self.augment_on_the_fly and len(forced) == 0:
                worker = get_worker_info()
                base_seed = worker.seed if worker is not None else self.augment_seed
                sample_seed = (int(base_seed) + int(idx) * 1000003) & 0xFFFFFFFF
                rng = np.random.default_rng(sample_seed)
                poses = _augment_poses_on_the_fly(
                    poses=poses,
                    rng=rng,
                    augment_prob=self.augment_prob,
                    max_ops=self.augment_max_ops,
                    op_probs=self.augment_op_probs,
                    ranges=self.augment_ranges,
                )
            poses = normalize_sequence(poses)
            poses = add_velocity(poses)
            poses = temporal_resize(poses, self.seq_len)
            t, j, d = poses.shape
            poses = poses.reshape(t, j * d)
            x = torch.from_numpy(poses.astype(np.float32))
            y = self.label_to_idx[ex.label]
            return x, y

        variants = self._variants_for_uid(uid)
        n_var = len(variants)
        worker = get_worker_info()
        base_seed = worker.seed if worker is not None else self.augment_seed
        sample_seed = (int(base_seed) + int(idx) * 1000003) & 0xFFFFFFFF
        rng = np.random.default_rng(sample_seed)

        if self.dataset_split in ("val", "test"):
            # Val/test: sin data augmentation (solo clip original normalizado).
            pass
        else:
            # train: primero determinista (opcional), luego aleatorio
            if self.use_deterministic_in_train and n_var > 0 and rng.random() < self.train_deterministic_prob:
                vidx = int(rng.integers(0, n_var))
                spec = variants[vidx]
                det_seed = (sample_seed ^ (vidx * 7919)) & 0xFFFFFFFF
                rng_det = np.random.default_rng(int(det_seed))
                poses = _apply_deterministic_ops(poses, spec.get("ops", []), rng=rng_det)
            if self.augment_on_the_fly:
                poses = _augment_poses_on_the_fly(
                    poses=poses,
                    rng=rng,
                    augment_prob=self.augment_prob,
                    max_ops=self.augment_max_ops,
                    op_probs=self.augment_op_probs,
                    ranges=self.augment_ranges,
                )

        poses = normalize_sequence(poses)
        poses = add_velocity(poses)  # [T, J, 4]
        poses = temporal_resize(poses, self.seq_len)  # [seq_len, J, 4]
        t, j, d = poses.shape
        poses = poses.reshape(t, j * d)  # [seq_len, F]
        x = torch.from_numpy(poses.astype(np.float32))  # [seq_len, F]
        y = self.label_to_idx[ex.label]
        return x, y


class PoseTCNClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] -> [B, F, T]
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class PoseResTCNClassifier(nn.Module):
    """
    TCN residual más profunda:
      - Varios bloques Conv1d + ReLU + Dropout con conexiones residuales.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] -> [B, F, T]
        x = x.permute(0, 2, 1)
        h = self.in_proj(x)
        for block in self.blocks:
            residual = h
            h = block(h)
            h = h + residual
        h = self.pool(h).squeeze(-1)
        return self.fc(h)


class PoseDilatedTCNClassifier(nn.Module):
    """
    TCN con convoluciones dilatadas para captar dependencias largas en el tiempo.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        layers = []
        dilation = 1
        for _ in range(num_layers):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
            dilation *= 2
        self.layers = nn.ModuleList(layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] -> [B, F, T]
        x = x.permute(0, 2, 1)
        h = self.in_proj(x)
        for layer in self.layers:
            h = h + layer(h)
        h = self.pool(h).squeeze(-1)
        return self.fc(h)

class PoseLSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        out, _ = self.lstm(x)
        # Usamos el último estado temporal
        last = out[:, -1, :]  # [B, 2*hidden]
        return self.fc(last)


class PoseTransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        b, t, _ = x.shape
        x = self.input_proj(x)  # [B, T, d_model]
        cls_tokens = self.cls_token.expand(b, 1, -1)  # [B, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+T, d_model]
        out = self.encoder(x)  # [B, 1+T, d_model]
        cls = out[:, 0, :]  # [B, d_model]
        return self.fc(cls)


class PoseSTGCNClassifier(nn.Module):
    """
    Versión muy simplificada de ST-GCN:
      - Reconstruye [B, T, J, F] a partir de [B, T, F*J]
      - Aplica una convolución de grafo fija sobre los joints
      - Luego una TCN 1D sobre el tiempo
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        # Asumimos 4 features por joint (x, y, dx, dy)
        if input_dim % 4 != 0:
            raise ValueError(f"PoseSTGCNClassifier espera input_dim múltiplo de 4, recibido {input_dim}")
        self.num_joints = input_dim // 4
        feat_per_joint = 4

        # Inicial: proyección por joint
        self.joint_mlp = nn.Linear(feat_per_joint, hidden_dim)

        # Adjacencia fija muy sencilla: cada joint conectado a sí mismo y vecinos inmediatos (cadena)
        A = torch.eye(self.num_joints)
        for j in range(self.num_joints - 1):
            A[j, j + 1] = 1.0
            A[j + 1, j] = 1.0
        # Normalización por grado
        deg = A.sum(dim=1, keepdim=True).clamp(min=1.0)
        A = A / deg
        self.register_buffer("A", A)  # [J, J]

        # TCN temporal después del grafo: trabajamos sobre canales=hidden_dim*num_joints
        tcn_input_dim = hidden_dim * self.num_joints
        self.tcn = nn.Sequential(
            nn.Conv1d(tcn_input_dim, tcn_input_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(tcn_input_dim, tcn_input_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(tcn_input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] con F = J*4
        b, t, f = x.shape
        j = self.num_joints
        x = x.view(b, t, j, 4)  # [B, T, J, 4]

        # Proyección por joint
        x = self.joint_mlp(x)  # [B, T, J, H]

        # Grafo: para cada frame aplicamos A sobre la dimensión de joints
        # x_g[b, t, j, h] = sum_k A[j, k] * x[b, t, k, h]
        x = x.permute(0, 1, 3, 2)  # [B, T, H, J]
        x = torch.matmul(x, self.A.T)  # [B, T, H, J]
        x = x.permute(0, 1, 3, 2)  # [B, T, J, H]

        # Aplanar joints en canales y aplicar TCN temporal
        x = x.reshape(b, t, -1)  # [B, T, J*H]
        x = x.permute(0, 2, 1)   # [B, C=J*H, T]
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)  # [B, C]
        return self.fc(x)


class PoseCNN2DClassifier(nn.Module):
    """
    CNN 2D sobre "imágenes" de poses:
      - Reconstruye [B, T, J, 4] a partir de [B, T, F*J]
      - Forma un mapa [B, 4, T, J] (canales = x,y,dx,dy)
      - Aplica Conv2D + pooling y clasificador final
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        if input_dim % 4 != 0:
            raise ValueError(f"PoseCNN2DClassifier espera input_dim múltiplo de 4, recibido {input_dim}")
        self.num_joints = input_dim // 4

        self.cnn = nn.Sequential(
            nn.Conv2d(4, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim),
            nn.Dropout2d(dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] con F = J*4
        b, t, f = x.shape
        j = self.num_joints
        x = x.view(b, t, j, 4)        # [B, T, J, 4]
        x = x.permute(0, 3, 1, 2)     # [B, 4, T, J]
        x = self.cnn(x)               # [B, C, T, J]
        x = self.pool(x).view(b, -1)  # [B, C]
        return self.fc(x)


class PoseJointAttnClassifier(nn.Module):
    """
    Modelo con atención por articulación + atención temporal:
      - Reconstruye [B, T, J, 4] a partir de [B, T, F*J]
      - Para cada frame, aplica un pequeño TransformerEncoder sobre los J joints (tokens = joints)
      - Obtiene un embedding por frame (media sobre joints)
      - Luego aplica un TransformerEncoder temporal sobre la secuencia de frames
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        joint_d_model: int = 64,
        temporal_d_model: int = 128,
        joint_layers: int = 1,
        temporal_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_dim % 4 != 0:
            raise ValueError(f"PoseJointAttnClassifier espera input_dim múltiplo de 4, recibido {input_dim}")
        self.num_joints = input_dim // 4

        # Proyección por joint
        self.joint_proj = nn.Linear(4, joint_d_model)
        joint_encoder_layer = nn.TransformerEncoderLayer(
            d_model=joint_d_model,
            nhead=min(nhead, joint_d_model),
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.joint_encoder = nn.TransformerEncoder(joint_encoder_layer, num_layers=joint_layers)

        # Proyección a espacio temporal
        self.frame_proj = nn.Linear(joint_d_model, temporal_d_model)

        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_d_model,
            nhead=min(nhead, temporal_d_model),
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_encoder_layer, num_layers=temporal_layers)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, temporal_d_model))
        self.fc = nn.Linear(temporal_d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] con F = J*4
        b, t, f = x.shape
        j = self.num_joints
        x = x.view(b, t, j, 4)  # [B, T, J, 4]

        # Atención por articulación (por frame)
        x = self.joint_proj(x)              # [B, T, J, Dj]
        x = x.view(b * t, j, -1)            # [B*T, J, Dj]
        x = self.joint_encoder(x)           # [B*T, J, Dj]
        x = x.mean(dim=1)                   # [B*T, Dj]  (media sobre joints)
        x = x.view(b, t, -1)                # [B, T, Dj]

        # Proyección a espacio temporal y Transformer temporal
        x = self.frame_proj(x)              # [B, T, Dt]
        cls_tokens = self.cls_token.expand(b, 1, -1)  # [B, 1, Dt]
        x = torch.cat([cls_tokens, x], dim=1)         # [B, 1+T, Dt]
        x = self.temporal_encoder(x)                  # [B, 1+T, Dt]
        cls = x[:, 0, :]                              # [B, Dt]
        return self.fc(cls)


class PoseTCNLSTMClassifier(nn.Module):
    """
    Híbrido TCN + BiLSTM:
      - TCN (Conv1d temporal) extrae features locales.
      - BiLSTM sobre la secuencia de features.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        tcn_hidden_dim: int = 128,
        tcn_layers: int = 2,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Conv1d(input_dim, tcn_hidden_dim, kernel_size=1)

        tcn_blocks = []
        for _ in range(tcn_layers):
            tcn_blocks.append(
                nn.Sequential(
                    nn.Conv1d(tcn_hidden_dim, tcn_hidden_dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        self.tcn_blocks = nn.ModuleList(tcn_blocks)

        self.lstm = nn.LSTM(
            input_size=tcn_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.fc = nn.Linear(lstm_hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        x = x.permute(0, 2, 1)  # [B, F, T]
        h = self.in_proj(x)     # [B, C, T]
        for block in self.tcn_blocks:
            h = h + block(h)
        h = h.permute(0, 2, 1)  # [B, T, C]
        out, _ = self.lstm(h)   # [B, T, 2*H]
        last = out[:, -1, :]
        return self.fc(last)


class PoseGRUClassifier(nn.Module):
    """
    Variante GRU bidireccional (más ligera que LSTM en cómputo/memoria).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)   # [B, T, 2*H]
        last = out[:, -1, :]
        return self.fc(last)


class PoseGRUAttnClassifier(nn.Module):
    """
    GRU bidireccional + atención temporal:
      - BiGRU extrae features temporales
      - Atención aprende a ponderar frames relevantes para la decisión
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        attn_dim = hidden_dim * 2
        self.attn = nn.Sequential(
            nn.Linear(attn_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )
        self.fc = nn.Linear(attn_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        out, _ = self.gru(x)  # [B, T, 2H]
        scores = self.attn(out).squeeze(-1)  # [B, T]
        weights = torch.softmax(scores, dim=1)  # [B, T]
        context = torch.sum(out * weights.unsqueeze(-1), dim=1)  # [B, 2H]
        return self.fc(context)


class PoseTCNGRUClassifier(nn.Module):
    """
    Híbrido TCN + BiGRU:
      - TCN captura patrones locales
      - GRU captura dependencias temporales con menor coste que LSTM
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        tcn_hidden_dim: int = 128,
        tcn_layers: int = 2,
        gru_hidden_dim: int = 128,
        gru_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Conv1d(input_dim, tcn_hidden_dim, kernel_size=1)
        blocks = []
        for _ in range(tcn_layers):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(tcn_hidden_dim, tcn_hidden_dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        self.tcn_blocks = nn.ModuleList(blocks)
        self.gru = nn.GRU(
            input_size=tcn_hidden_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.fc = nn.Linear(gru_hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.permute(0, 2, 1)   # [B, F, T]
        h = self.in_proj(h)
        for block in self.tcn_blocks:
            h = h + block(h)
        h = h.permute(0, 2, 1)   # [B, T, C]
        out, _ = self.gru(h)
        last = out[:, -1, :]
        return self.fc(last)


class PoseConformerLiteClassifier(nn.Module):
    """
    Conformer temporal ligero:
      - Bloque MHSA (TransformerEncoderLayer)
      - Bloque conv depthwise temporal
      - Residuales sobre secuencia
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        conv_kernel: int = 7,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.attn_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ]
        )
        padding = conv_kernel // 2
        self.conv_pw1 = nn.ModuleList([nn.Conv1d(d_model, 2 * d_model, kernel_size=1) for _ in range(num_layers)])
        self.conv_dw = nn.ModuleList(
            [
                nn.Conv1d(
                    2 * d_model,
                    2 * d_model,
                    kernel_size=conv_kernel,
                    padding=padding,
                    groups=2 * d_model,
                )
                for _ in range(num_layers)
            ]
        )
        self.conv_pw2 = nn.ModuleList([nn.Conv1d(2 * d_model, d_model, kernel_size=1) for _ in range(num_layers)])
        self.conv_act = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)  # [B, T, D]
        for i, attn in enumerate(self.attn_layers):
            h = h + attn(h)
            hc = h.transpose(1, 2)  # [B, D, T]
            hc = self.conv_pw1[i](hc)
            hc = self.conv_act(hc)
            hc = self.conv_dw[i](hc)
            hc = self.conv_act(hc)
            hc = self.conv_pw2[i](hc)
            hc = hc.transpose(1, 2)  # [B, T, D]
            h = self.norm(h + hc)
        pooled = h.mean(dim=1)  # [B, D]
        return self.fc(pooled)


class PoseMSTCNClassifier(nn.Module):
    """
    Multi-Scale TCN:
      - ramas paralelas con distintos kernels/dilataciones
      - fusión + residual para capturar acciones rápidas/lentas
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "b1": nn.Sequential(
                            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                        ),
                        "b2": nn.Sequential(
                            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=4, dilation=2),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                        ),
                        "b3": nn.Sequential(
                            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=9, dilation=3),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                        ),
                        "fuse": nn.Conv1d(hidden_dim * 3, hidden_dim, kernel_size=1),
                    }
                )
            )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x.permute(0, 2, 1))  # [B, C, T]
        for blk in self.blocks:
            residual = h
            m1 = blk["b1"](h)
            m2 = blk["b2"](h)
            m3 = blk["b3"](h)
            h = blk["fuse"](torch.cat([m1, m2, m3], dim=1))
            h = h + residual
        h = self.pool(h).squeeze(-1)
        return self.fc(h)


class PoseTCNAttnClassifier(nn.Module):
    """
    TCN + atención temporal:
      - TCN extrae features locales
      - atención pondera frames críticos de la acción
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        self.layers = nn.ModuleList(layers)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x.permute(0, 2, 1))  # [B, C, T]
        for layer in self.layers:
            h = h + layer(h)
        ht = h.permute(0, 2, 1)  # [B, T, C]
        scores = self.attn(ht).squeeze(-1)  # [B, T]
        w = torch.softmax(scores, dim=1)
        context = torch.sum(ht * w.unsqueeze(-1), dim=1)  # [B, C]
        return self.fc(context)


class PoseGATTCNClassifier(nn.Module):
    """
    GAT espacial ligero + TCN temporal:
      - atención entre joints por frame (no adyacencia fija)
      - TCN sobre secuencia temporal del embedding espacial
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        tcn_hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_dim % 4 != 0:
            raise ValueError(f"PoseGATTCNClassifier espera input_dim múltiplo de 4, recibido {input_dim}")
        self.num_joints = input_dim // 4
        self.joint_proj = nn.Linear(4, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.tcn_in = nn.Conv1d(hidden_dim * self.num_joints, tcn_hidden_dim, kernel_size=1)
        self.tcn = nn.Sequential(
            nn.Conv1d(tcn_hidden_dim, tcn_hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(tcn_hidden_dim, tcn_hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(tcn_hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, f = x.shape
        j = self.num_joints
        h = x.view(b, t, j, 4)  # [B, T, J, 4]
        h = self.joint_proj(h)  # [B, T, J, H]
        h_bt = h.reshape(b * t, j, -1)  # [B*T, J, H]
        q = self.q_proj(h_bt)
        k = self.k_proj(h_bt)
        v = self.v_proj(h_bt)
        attn = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / (q.shape[-1] ** 0.5), dim=-1)  # [B*T, J, J]
        h_bt = self.out_proj(torch.matmul(attn, v)) + h_bt  # [B*T, J, H]
        h = h_bt.reshape(b, t, j, -1).reshape(b, t, -1)  # [B, T, J*H]
        h = self.tcn_in(h.permute(0, 2, 1))  # [B, C, T]
        h = self.tcn(h)
        h = self.pool(h).squeeze(-1)
        return self.fc(h)


def split_examples(
    examples: List[PoseExample],
    seed: int = SEED,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> Tuple[List[PoseExample], List[PoseExample], List[PoseExample]]:
    # Split determinista y estable entre ejecuciones/experimentos:
    # 1) orden base estable por UID
    # 2) shuffle con RNG local semillado
    tr = float(max(0.0, min(1.0, train_ratio)))
    vr = float(max(0.0, min(1.0, val_ratio)))
    if tr + vr > 1.0:
        raise ValueError(f"train_ratio+val_ratio debe ser <= 1.0 (got {tr}+{vr})")
    ordered = sorted(examples, key=_example_uid)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n = len(ordered)
    n_train = int(n * tr)
    n_val = int(n * vr)
    train = ordered[:n_train]
    val = ordered[n_train:n_train + n_val]
    test = ordered[n_train + n_val:]
    return train, val, test


def verify_split_uid_disjoint(split_uids: Dict[str, List[str]]) -> None:
    """Garantiza que ningún UID aparece en más de un split (train/val/test)."""
    seen: Dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for uid in split_uids.get(split_name, []):
            u = str(uid)
            if u in seen:
                raise ValueError(
                    f"Fuga de split: UID {u!r} en {seen[u]} y {split_name}"
                )
            seen[u] = split_name


def unique_uids_from_examples(examples: List[PoseExample]) -> List[str]:
    return sorted({_example_uid(ex) for ex in examples})


def split_uids_from_example_lists(
    train_ex: List[PoseExample],
    val_ex: List[PoseExample],
    test_ex: List[PoseExample],
) -> Dict[str, List[str]]:
    split_uids = {
        "train": unique_uids_from_examples(train_ex),
        "val": unique_uids_from_examples(val_ex),
        "test": unique_uids_from_examples(test_ex),
    }
    verify_split_uid_disjoint(split_uids)
    return split_uids


def split_examples_by_uid_manifest(
    examples: List[PoseExample],
    split_uids: Dict[str, List[str]],
) -> Tuple[List[PoseExample], List[PoseExample], List[PoseExample]]:
    """
    Reparte ejemplos según UID fijado en el plan (clips reales, antes de augment).
    Ignora ejemplos cuyo UID no está en el plan.
    """
    verify_split_uid_disjoint(split_uids)
    train_set = set(split_uids.get("train", []))
    val_set = set(split_uids.get("val", []))
    test_set = set(split_uids.get("test", []))
    train: List[PoseExample] = []
    val: List[PoseExample] = []
    test: List[PoseExample] = []
    for ex in examples:
        uid = _example_uid(ex)
        if uid in train_set:
            train.append(ex)
        elif uid in val_set:
            val.append(ex)
        elif uid in test_set:
            test.append(ex)
    return train, val, test


def split_examples_stratified_by_uid(
    examples: List[PoseExample],
    seed: int = SEED,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    stratify_key=None,
) -> Tuple[List[PoseExample], List[PoseExample], List[PoseExample]]:
    """
    Split determinista a nivel UID (clip): todos los ejemplos del mismo clip van al mismo split.
    Estratifica por categoría de carpeta para repartir mejor val/test por clase.
    """
    if stratify_key is None:
        stratify_key = _example_folder_category
    tr = float(max(0.0, min(1.0, train_ratio)))
    vr = float(max(0.0, min(1.0, val_ratio)))
    if tr + vr > 1.0:
        raise ValueError(f"train_ratio+val_ratio debe ser <= 1.0 (got {tr}+{vr})")

    uid_to_exs: Dict[str, List[PoseExample]] = {}
    uid_to_cat: Dict[str, int] = {}
    for ex in examples:
        uid = _example_uid(ex)
        uid_to_exs.setdefault(uid, []).append(ex)
        uid_to_cat[uid] = int(stratify_key(ex))

    by_cat: Dict[int, List[str]] = {}
    for uid, cat in uid_to_cat.items():
        by_cat.setdefault(cat, []).append(uid)

    rng = random.Random(seed)
    train_uids: List[str] = []
    val_uids: List[str] = []
    test_uids: List[str] = []
    for _cat, uids in sorted(by_cat.items(), key=lambda x: int(x[0])):
        ordered = sorted(uids)
        rng.shuffle(ordered)
        n = len(ordered)
        n_train = int(n * tr)
        n_val = int(n * vr)
        train_uids.extend(ordered[:n_train])
        val_uids.extend(ordered[n_train : n_train + n_val])
        test_uids.extend(ordered[n_train + n_val :])

    def _collect(uids: List[str]) -> List[PoseExample]:
        out: List[PoseExample] = []
        for uid in uids:
            out.extend(uid_to_exs[uid])
        return out

    train = _collect(train_uids)
    val = _collect(val_uids)
    test = _collect(test_uids)
    split_uids_from_example_lists(train, val, test)
    return train, val, test


def assert_no_uid_leak_between_splits(
    train_ex: List[PoseExample],
    val_ex: List[PoseExample],
    test_ex: List[PoseExample],
) -> None:
    train_uids = {_example_uid(ex) for ex in train_ex}
    val_uids = {_example_uid(ex) for ex in val_ex}
    test_uids = {_example_uid(ex) for ex in test_ex}
    leak_tv = train_uids & val_uids
    leak_tt = train_uids & test_uids
    leak_vt = val_uids & test_uids
    if leak_tv or leak_tt or leak_vt:
        raise ValueError(
            "Fuga de UID entre splits: "
            f"train∩val={len(leak_tv)} train∩test={len(leak_tt)} val∩test={len(leak_vt)}"
        )


def load_training_plan_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "split_uids" not in data:
        raise ValueError(f"training_plan inválido (falta split_uids): {path}")
    verify_split_uid_disjoint(data["split_uids"])
    return data


def build_plan_stats_by_category(
    train_ex: List[PoseExample],
    val_ex: List[PoseExample],
    test_ex: List[PoseExample],
    category_aug_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Clips reales vs filas sintéticas (augment) por categoría de carpeta."""
    include_identity = bool(category_aug_cfg.get("include_identity", True))
    train_counts = count_examples_by_folder_category(train_ex)
    val_counts = count_examples_by_folder_category(val_ex)
    test_counts = count_examples_by_folder_category(test_ex)
    rows_detail = summarize_category_aug_on_train(train_counts, category_aug_cfg)

    by_cat: Dict[str, Dict[str, int]] = {}
    cats = sorted(
        set(train_counts) | set(val_counts) | set(test_counts),
        key=lambda x: int(x),
    )
    total_real_train = 0
    total_synthetic_train = 0
    total_rows_train = 0
    for cat in cats:
        clips = int(train_counts.get(cat, 0))
        rows = int(rows_detail.get(cat, {}).get("train_rows", clips))
        synthetic = max(0, rows - clips)
        by_cat[str(cat)] = {
            "clips_real_train": clips,
            "clips_real_val": int(val_counts.get(cat, 0)),
            "clips_real_test": int(test_counts.get(cat, 0)),
            "aug_per_clip": int(rows_detail.get(cat, {}).get("aug_per_clip", 0)),
            "rows_synthetic_train": synthetic,
            "rows_total_train": rows,
        }
        total_real_train += clips
        total_synthetic_train += synthetic
        total_rows_train += rows

    return {
        "by_category": by_cat,
        "totals": {
            "clips_real_train": total_real_train,
            "clips_real_val": sum(val_counts.values()),
            "clips_real_test": sum(test_counts.values()),
            "rows_synthetic_train": total_synthetic_train,
            "rows_total_train": total_rows_train,
            "include_identity_in_rows": include_identity,
        },
    }


def max_category_augment_ops_available() -> int:
    return len(CATEGORY_AUGMENT_RECIPE_BANK)


def build_label_mapping(examples: List[PoseExample]) -> Dict[int, int]:
    labels = sorted({ex.label for ex in examples})
    return {lab: i for i, lab in enumerate(labels)}


def make_binary_examples(
    examples: List[PoseExample],
    positive_class: int = 6,
) -> List[PoseExample]:
    """
    Construye una lista de ejemplos binarios:
      - label = 1 si la clase original == positive_class
      - label = 0 en caso contrario
    """
    binary_examples: List[PoseExample] = []
    for ex in examples:
        bin_label = 1 if ex.label == positive_class else 0
        binary_examples.append(
            PoseExample(
                pose_path=ex.pose_path,
                label=bin_label,
                track_id=ex.track_id,
                clip_name=ex.clip_name,
                category_str=ex.category_str,
                valid_mask_path=getattr(ex, "valid_mask_path", None),
                users_in_clip=getattr(ex, "users_in_clip", 1),
                action_label=_example_action_label(ex),
                forced_ops=getattr(ex, "forced_ops", None),
            )
        )
    return binary_examples


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
    return total_loss / max(total, 1)


def _classification_loss_per_sample(
    logits: torch.Tensor,
    y: torch.Tensor,
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
    asym_gamma_neg: float = 4.0,
    asym_gamma_pos: float = 1.0,
) -> torch.Tensor:
    loss_type = str(loss_type).lower()
    ce = F.cross_entropy(logits, y, reduction="none")
    if loss_type == "ce":
        return ce
    probs = torch.softmax(logits, dim=1)
    pt = probs.gather(1, y.view(-1, 1)).squeeze(1).clamp(min=1e-6, max=1.0 - 1e-6)
    if loss_type == "focal":
        mod = torch.pow(1.0 - pt, float(max(0.0, focal_gamma)))
        return ce * mod
    if loss_type == "asymmetric":
        if logits.shape[1] == 2:
            yb = (y > 0).float()
            p_pos = probs[:, 1].clamp(min=1e-6, max=1.0 - 1e-6)
            p_t = yb * p_pos + (1.0 - yb) * (1.0 - p_pos)
            gam = yb * float(max(0.0, asym_gamma_pos)) + (1.0 - yb) * float(max(0.0, asym_gamma_neg))
            bce = -(yb * torch.log(p_pos) + (1.0 - yb) * torch.log(1.0 - p_pos))
            return bce * torch.pow(1.0 - p_t, gam)
        # fallback multiclase
        mod = torch.pow(1.0 - pt, float(max(0.0, focal_gamma)))
        return ce * mod
    return ce


def _mixstyle_features(x: torch.Tensor, p: float = 0.0, alpha: float = 0.3) -> torch.Tensor:
    # MixStyle simple en espacio de features temporales [B,T,F] para robustez de dominio/cámara.
    if p <= 0.0 or x.dim() != 3:
        return x
    if torch.rand(1, device=x.device).item() >= p or x.shape[0] < 2:
        return x
    eps = 1e-6
    mu = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True).clamp(min=eps)
    x_norm = (x - mu) / std
    perm = torch.randperm(x.size(0), device=x.device)
    mu2 = mu[perm]
    std2 = std[perm]
    lam = torch.distributions.Beta(alpha, alpha).sample((x.size(0), 1, 1)).to(x.device)
    mu_mix = lam * mu + (1.0 - lam) * mu2
    std_mix = lam * std + (1.0 - lam) * std2
    return x_norm * std_mix + mu_mix


def _ssl_view_jitter(x: torch.Tensor, noise_std: float = 0.01, drop_prob: float = 0.05) -> torch.Tensor:
    out = x
    if noise_std > 0.0:
        out = out + torch.randn_like(out) * float(noise_std)
    if drop_prob > 0.0:
        keep = (torch.rand_like(out[..., :1]) >= float(drop_prob)).float()
        out = out * keep
    return out


def train_one_epoch_advanced(
    model,
    loader,
    optimizer,
    device,
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
    asym_gamma_neg: float = 4.0,
    asym_gamma_pos: float = 1.0,
    hard_negative_mining: bool = False,
    hard_negative_topk_frac: float = 0.2,
    hard_negative_weight: float = 1.5,
    mixstyle_prob: float = 0.0,
    mixstyle_alpha: float = 0.3,
    ssl_consistency_weight: float = 0.0,
    ssl_noise_std: float = 0.01,
    ssl_drop_prob: float = 0.05,
    supervised_weight: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        x = _mixstyle_features(x, p=float(max(0.0, mixstyle_prob)), alpha=float(max(1e-3, mixstyle_alpha)))
        optimizer.zero_grad()
        logits = model(x)
        per_sample = _classification_loss_per_sample(
            logits,
            y,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            asym_gamma_neg=asym_gamma_neg,
            asym_gamma_pos=asym_gamma_pos,
        )
        if hard_negative_mining and logits.shape[1] == 2:
            weights = torch.ones_like(per_sample)
            neg_mask = y == 0
            neg_idx = torch.where(neg_mask)[0]
            if neg_idx.numel() > 0:
                p_pos = torch.softmax(logits, dim=1)[:, 1]
                k = max(1, int(round(float(max(0.0, min(1.0, hard_negative_topk_frac))) * neg_idx.numel())))
                hardest_local = torch.topk(p_pos[neg_idx], k=min(k, neg_idx.numel()), largest=True).indices
                hard_idx = neg_idx[hardest_local]
                weights[hard_idx] = float(max(1.0, hard_negative_weight))
            per_sample = per_sample * weights
        sup_loss = per_sample.mean()
        loss = sup_loss * float(max(0.0, supervised_weight))
        if ssl_consistency_weight > 0.0:
            x_ssl = _ssl_view_jitter(x, noise_std=ssl_noise_std, drop_prob=ssl_drop_prob)
            logits_ssl = model(x_ssl)
            p_detached = torch.softmax(logits.detach(), dim=1)
            ssl_loss = F.kl_div(F.log_softmax(logits_ssl, dim=1), p_detached, reduction="batchmean")
            loss = loss + float(ssl_consistency_weight) * ssl_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


@torch.no_grad()
def _binary_metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int = 1,
) -> Dict[str, float]:
    tp = int(((y_true == pos_label) & (y_pred == pos_label)).sum())
    fn = int(((y_true == pos_label) & (y_pred != pos_label)).sum())
    fp = int(((y_true != pos_label) & (y_pred == pos_label)).sum())
    tn = int(((y_true != pos_label) & (y_pred != pos_label)).sum())
    n_pos = int((y_true == pos_label).sum())
    n_neg = int((y_true != pos_label).sum())
    n = max(len(y_true), 1)
    acc = (tp + tn) / n
    recall = tp / max(n_pos, 1)
    precision = tp / max(tp + fp, 1)
    fp_rate = fp / max(n_neg, 1)
    return {
        "accuracy_pct": float(100.0 * acc),
        "recall_robbery_pct": float(100.0 * recall),
        "precision_robbery_pct": float(100.0 * precision),
        "false_positive_rate_pct": float(100.0 * fp_rate),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support_pos": n_pos,
        "support_neg": n_neg,
    }


def format_binary_metrics_line(binary_metrics: Dict[str, Any], prefix: str = "") -> str:
    parts = []
    for key, label in (
        ("softmax_argmax", "argmax"),
        ("softmax_threshold", "softmax@thr"),
        ("logit_margin", "logit@margin"),
    ):
        block = binary_metrics.get(key)
        if not block:
            continue
        parts.append(
            f"{label}: acc={block['accuracy_pct']:.1f}% "
            f"rec_robo={block['recall_robbery_pct']:.1f}% "
            f"FP={block['false_positive_rate_pct']:.1f}%"
        )
    return (prefix + " | ".join(parts)) if parts else ""


@torch.no_grad()
def evaluate_with_metrics(
    model,
    loader,
    criterion,
    device,
    num_classes: int,
    task: str = "multiclass",
    binary_softmax_threshold: float = DEFAULT_BINARY_SOFTMAX_THRESHOLD,
    binary_logit_margin: float = DEFAULT_BINARY_LOGIT_MARGIN,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Evalúa en un loader y devuelve:
      - pérdida media
      - accuracy (softmax argmax)
      - métricas detalladas: matriz de confusión, precision/recall/F1 por clase,
        macro/weighted F1, top-3 accuracy, % acierto por clase (multiclass),
        y comparativa softmax-argmax / softmax-umbral / logits (binario).
    """
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    conf_mat = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    top3_correct = 0
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        total += x.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

        for yt, yp in zip(y.tolist(), preds.tolist()):
            if 0 <= yt < num_classes and 0 <= yp < num_classes:
                conf_mat[yt][yp] += 1

        if logits.size(1) >= 3:
            top3 = logits.topk(3, dim=1).indices
            for yt, topk in zip(y.tolist(), top3.tolist()):
                if yt in topk:
                    top3_correct += 1

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)

    per_class = {}
    supports = []
    f1s = []

    for c in range(num_classes):
        tp = conf_mat[c][c]
        fn = sum(conf_mat[c][j] for j in range(num_classes)) - tp
        fp = sum(conf_mat[i][c] for i in range(num_classes)) - tp
        support = tp + fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0

        per_class[c] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": int(support),
            "accuracy_pct": float(100.0 * tp / support) if support > 0 else None,
        }

        supports.append(support)
        f1s.append(f1)

    total_support = sum(supports) if supports else 0
    if total_support > 0:
        macro_f1 = float(sum(f1s) / max(len(f1s), 1))
        weighted_f1 = float(
            sum(f * s for f, s in zip(f1s, supports)) / total_support
        )
    else:
        macro_f1 = 0.0
        weighted_f1 = 0.0

    top3_acc = top3_correct / max(total, 1) if total > 0 else 0.0

    metrics: Dict[str, Any] = {
        "confusion_matrix": conf_mat,
        "per_class": per_class,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "top3_acc": float(top3_acc),
    }

    if num_classes == 2 and all_logits:
        logits_cat = torch.cat(all_logits, dim=0)
        y_np = torch.cat(all_labels, dim=0).numpy()
        probs_pos = F.softmax(logits_cat, dim=1)[:, 1].numpy()
        margin = (logits_cat[:, 1] - logits_cat[:, 0]).numpy()
        preds_argmax = logits_cat.argmax(dim=1).numpy()
        preds_softmax_thr = (probs_pos >= float(binary_softmax_threshold)).astype(np.int64)
        preds_logit_thr = (margin >= float(binary_logit_margin)).astype(np.int64)
        metrics["binary"] = {
            "softmax_threshold_value": float(binary_softmax_threshold),
            "logit_margin_value": float(binary_logit_margin),
            "softmax_argmax": _binary_metrics_from_predictions(y_np, preds_argmax, pos_label=1),
            "softmax_threshold": _binary_metrics_from_predictions(
                y_np, preds_softmax_thr, pos_label=1
            ),
            "logit_margin": _binary_metrics_from_predictions(
                y_np, preds_logit_thr, pos_label=1
            ),
        }

    if task == "multiclass" and num_classes > 2:
        metrics["per_class_accuracy_pct"] = {
            str(c): per_class[c]["accuracy_pct"] for c in range(num_classes)
        }

    return avg_loss, acc, metrics


def build_datasets_and_loaders(
    seq_len: int,
    batch_size: int,
    pose_source: str,
    num_workers: int = 4,
    task: str = "multiclass",
    positive_class: int = 6,
    balanced: bool = False,
    single_user_only: bool = False,
    min_clip_seconds: float = MIN_CLIP_SECONDS,
    min_valid_frames: int = MIN_VALID_FRAMES,
    min_valid_pct: float = MIN_VALID_PCT,
    max_occlusion_ratio: float = MAX_OCCLUSION_RATIO,
    augment_on_the_fly: bool = False,
    augment_config_path: Path = AUGMENT_CONFIG_PATH,
    augment_profile: str = AUGMENT_PROFILE_DEFAULT,
    augment_prob: float = AUGMENT_PROB,
    augment_max_ops: int = AUGMENT_MAX_OPS,
    augment_seed: int = SEED,
    maintain_class_ratio: bool = False,
    target_neg_pos_ratio: Optional[float] = None,
    train_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
    max_deterministic_variants: int = MAX_DETERMINISTIC_VARIANTS,
    train_deterministic_prob: float = TRAIN_DETERMINISTIC_PROB,
    use_deterministic_in_train: bool = True,
    manifest_cache_dir: Optional[Path] = None,
    manifest_variant_set: str = MANIFEST_VARIANT_SET_DEFAULT,
    extra_manifest_views_per_clip: int = 0,
    category_aug_config_path: Optional[Path] = None,
    use_category_augmentation: Optional[bool] = None,
    training_plan_path: Optional[Path] = None,
    stratified_split: bool = True,
    hard_negative_uid_set: Optional[set] = None,
    hard_negative_uid_weight: float = 3.0,
) -> Tuple[Dict[str, DataLoader], int, Dict[int, int], Dict[str, Any]]:
    print(f"Recolectando ejemplos desde data_result... (pose_source='{pose_source}')")
    examples = collect_examples(
        pose_source=pose_source,
        single_user_only=single_user_only,
        min_clip_seconds=float(min_clip_seconds),
        min_valid_frames=int(min_valid_frames),
        min_valid_pct=float(min_valid_pct),
        max_occlusion_ratio=float(max_occlusion_ratio),
    )
    print(f"Ejemplos totales (tras filtrado): {len(examples)}")

    training_plan: Optional[Dict[str, Any]] = None
    if training_plan_path is not None:
        training_plan = load_training_plan_json(training_plan_path)
        print(f"[TRAINING-PLAN] Cargado: {training_plan_path}")
        plan_task = training_plan.get("task", "multiclass")
        if plan_task != task:
            raise ValueError(
                f"task={task!r} no coincide con training_plan ({plan_task!r})"
            )
        if task == "binary" and int(training_plan.get("positive_class", positive_class)) != int(positive_class):
            raise ValueError(
                "positive_class del CLI no coincide con el training_plan"
            )

    if training_plan is not None and training_plan.get("class_map"):
        try:
            from .class_map_utils import apply_class_map_spec
        except ImportError:
            from class_map_utils import apply_class_map_spec  # type: ignore
        n_before = len(examples)
        examples = apply_class_map_spec(examples, training_plan["class_map"])
        print(
            f"[CLASS-MAP] {training_plan['class_map'].get('id')}: "
            f"{n_before} → {len(examples)} ejemplos"
        )

    if DEBUG_MODE:
        examples = examples[:DEBUG_MAX_EXAMPLES]
        print(f"[DEBUG] Usando solo {len(examples)} ejemplos para train/val/test")

    if task == "binary":
        print(f"[BINARIO] Usando clase positiva original: {positive_class}")
        examples = make_binary_examples(examples, positive_class=positive_class)

    n_ex = len(examples)
    if training_plan is not None:
        tr = float(training_plan["split_ratios"]["train"])
        vr = float(training_plan["split_ratios"]["val"])
        te = float(training_plan["split_ratios"]["test"])
        print(
            f"[SPLIT] Desde training_plan: {tr:.3f}/{vr:.3f}/{te:.3f} train/val/test"
        )
        train_ex, val_ex, test_ex = split_examples_by_uid_manifest(
            examples, training_plan["split_uids"]
        )
        plan_all = set(training_plan["split_uids"].get("train", []))
        plan_all |= set(training_plan["split_uids"].get("val", []))
        plan_all |= set(training_plan["split_uids"].get("test", []))
        loaded_uids = {_example_uid(ex) for ex in examples}
        missing = plan_all - loaded_uids
        extra = loaded_uids - plan_all
        if missing:
            print(f"[TRAINING-PLAN] Advertencia: {len(missing)} UIDs del plan no están en los datos actuales")
        if extra:
            print(f"[TRAINING-PLAN] {len(extra)} UIDs en datos sin asignación en plan (ignorados)")
    else:
        if (train_ratio is None) ^ (val_ratio is None):
            raise ValueError("Indica ambos --train-ratio y --val-ratio, o ninguno para usar la heurística suggest_split_ratios(N).")
        if train_ratio is None or val_ratio is None:
            tr, vr, te = suggest_split_ratios(n_ex)
            print(
                f"[SPLIT] Heurística suggest_split_ratios(N={n_ex}) = "
                f"{tr:.3f}/{vr:.3f}/{te:.3f} train/val/test (misma lógica que preflight; "
                "o pasa --train-ratio y --val-ratio para fijar a mano)."
            )
        else:
            tr = float(train_ratio)
            vr = float(val_ratio)
            if tr + vr > 1.0:
                raise ValueError(f"train_ratio+val_ratio debe ser <= 1.0 (got {tr}+{vr})")
            te = 1.0 - tr - vr
            sug_tr, sug_vr, sug_te = suggest_split_ratios(n_ex)
            print(
                f"[SPLIT] Ratios explícitos: {tr:.3f}/{vr:.3f}/{te:.3f} train/val/test | "
                f"heurística para N={n_ex} sería {sug_tr:.3f}/{sug_vr:.3f}/{sug_te:.3f}"
            )

        if stratified_split:
            train_ex, val_ex, test_ex = split_examples_stratified_by_uid(
                examples, seed=SEED, train_ratio=tr, val_ratio=vr
            )
            print("[SPLIT] Estratificado por categoría de carpeta (UID único por clip)")
        else:
            train_ex, val_ex, test_ex = split_examples(
                examples, seed=SEED, train_ratio=tr, val_ratio=vr
            )

    assert_no_uid_leak_between_splits(train_ex, val_ex, test_ex)
    split_uids_clips = split_uids_from_example_lists(train_ex, val_ex, test_ex)
    print(
        f"Split aplicado => Train: {len(train_ex)} | Val: {len(val_ex)} | Test: {len(test_ex)} "
        f"(UIDs únicos: {len(split_uids_clips['train'])}/"
        f"{len(split_uids_clips['val'])}/{len(split_uids_clips['test'])})"
    )

    train_unique_npy = len({ex.pose_path.resolve() for ex in train_ex})
    val_unique_npy = len({ex.pose_path.resolve() for ex in val_ex})
    test_unique_npy = len({ex.pose_path.resolve() for ex in test_ex})
    train_rows_before_expand = len(train_ex)

    aug_cfg = _load_augment_profile(augment_config_path, augment_profile)
    if not aug_cfg:
        aug_cfg = {
            "steps": {"rotate": 2.0, "scale": 2.0, "shift": 0.02, "speed": 0.15},
            "rotate": {"min": -15.0, "max": 15.0},
            "noise": {"sigma_cap": 0.006},
            "speed": {"min": AUGMENT_SPEED_FACTOR_LO, "max": AUGMENT_SPEED_FACTOR_HI},
        }
    rotate_cfg = aug_cfg.get("rotate", {}) if isinstance(aug_cfg, dict) else {}
    noise_cfg = aug_cfg.get("noise", {}) if isinstance(aug_cfg, dict) else {}
    speed_cfg = aug_cfg.get("speed", {}) if isinstance(aug_cfg, dict) else {}
    aug_ranges = {
        "rotate_degrees": (float(rotate_cfg.get("min", -15.0)), float(rotate_cfg.get("max", 15.0))),
        "scale_percentage": (95.0, 111.0),
        "shift_dx": (-0.06, 0.06),
        "shift_dy": (-0.06, 0.06),
        "noise_sigma_x": (0.0, float(noise_cfg.get("sigma_cap", 0.006))),
        "noise_sigma_y": (0.0, float(noise_cfg.get("sigma_cap", 0.006))),
        "speed_factor": (
            float(speed_cfg.get("min", AUGMENT_SPEED_FACTOR_LO)),
            float(speed_cfg.get("max", AUGMENT_SPEED_FACTOR_HI)),
        ),
    }

    cat_aug_path = Path(category_aug_config_path or DEFAULT_CATEGORY_AUGMENTATION_CONFIG_PATH)
    if training_plan is not None and training_plan.get("category_augmentation_config"):
        cat_aug_path = Path(training_plan["category_augmentation_config"])
    cat_aug_cfg = load_category_augmentation_config(cat_aug_path)
    if use_category_augmentation is False:
        cat_aug_cfg["enabled"] = False
    elif use_category_augmentation is True:
        cat_aug_cfg["enabled"] = True
    if _category_augment_is_active(cat_aug_cfg):
        print(f"[CATEGORY-AUG] Config: {cat_aug_path} | solo split train (val/test sin augmentación)")
        train_ex = expand_examples_with_category_augmentation(
            train_ex, cat_aug_cfg, augment_ranges=aug_ranges, seed=augment_seed
        )
        print(
            f"Tras expansión por categoría (solo train) => Train filas: {len(train_ex)} | "
            f"Val filas: {len(val_ex)} | Test filas: {len(test_ex)}"
        )
        assert_no_uid_leak_between_splits(train_ex, val_ex, test_ex)
    else:
        print(f"[CATEGORY-AUG] Desactivado (config {cat_aug_path}, enabled={cat_aug_cfg.get('enabled', True)})")

    if extra_manifest_views_per_clip > 0:
        if manifest_cache_dir is None:
            raise ValueError(
                "extra_manifest_views_per_clip>0 requiere --manifest-cache-dir con JSON validate_npy por UID."
            )
        train_ex = expand_examples_with_manifest_extra_views(
            train_ex,
            Path(manifest_cache_dir),
            manifest_variant_set,
            extra_manifest_views_per_clip,
        )
        print(
            f"Tras expansión manifest (solo train) => Train filas: {len(train_ex)} | "
            f"Val filas: {len(val_ex)} | Test (sin expandir): {len(test_ex)}"
        )

    label_to_idx = build_label_mapping(examples)
    num_classes = len(label_to_idx)
    print(f"Número de clases: {num_classes} | mapping: {label_to_idx}")

    det_specs = build_deterministic_variant_specs(
        aug_cfg if isinstance(aug_cfg, dict) else {},
        max_variants=max(1, int(max_deterministic_variants)),
    )
    print(
        f"[AUGMENT] Perfil={augment_profile} | variantes deterministas (rejilla validate_npy): {len(det_specs)}"
    )
    per_uid_map: Optional[Dict[str, List[Dict[str, Any]]]] = None
    manifest_hits = 0
    if manifest_cache_dir is not None:
        mdir = Path(manifest_cache_dir)
        per_uid_map = build_per_uid_variant_map(
            train_ex + val_ex + test_ex,
            mdir,
            variant_set=manifest_variant_set,
            verify_source_path=True,
        )
        manifest_hits = len(per_uid_map)
        print(
            f"[MANIFEST-CACHE] dir={mdir} | UIDs con manifest propio: {manifest_hits} | "
            f"variant_set={manifest_variant_set} | resto usa rejilla global"
        )
    tr_g, tr_mi, tr_mo = _summarize_forced_rows(train_ex)
    _print_training_pool_summary_once(
        train_unique_npy=train_unique_npy,
        train_rows=len(train_ex),
        train_rows_before_expand=train_rows_before_expand,
        extra_manifest_views_per_clip=int(extra_manifest_views_per_clip),
        val_unique_npy=val_unique_npy,
        val_rows=len(val_ex),
        test_unique_npy=test_unique_npy,
        test_rows=len(test_ex),
        n_det_grid_specs=len(det_specs),
        manifest_hits=int(manifest_hits),
        train_grid_only=tr_g,
        train_manifest_identity=tr_mi,
        train_manifest_ops=tr_mo,
        augment_on_the_fly=bool(augment_on_the_fly),
        augment_prob=float(augment_prob),
        use_deterministic_in_train=bool(use_deterministic_in_train),
        train_deterministic_prob=float(train_deterministic_prob),
    )
    train_ds = PoseDataset(
        train_ex,
        label_to_idx,
        seq_len,
        augment_on_the_fly=augment_on_the_fly,
        augment_prob=augment_prob,
        augment_max_ops=augment_max_ops,
        augment_op_probs={
            "mirror": 0.26,
            "rotate": 0.24,
            "scale": 0.14,
            "shift": 0.14,
            "noise": 0.08,
            "speed": 0.14,
        },
        augment_ranges=aug_ranges,
        augment_seed=augment_seed,
        dataset_split="train",
        deterministic_variants=det_specs,
        train_deterministic_prob=train_deterministic_prob,
        use_deterministic_in_train=use_deterministic_in_train,
        per_uid_variants=per_uid_map,
    )
    val_ds = PoseDataset(
        val_ex,
        label_to_idx,
        seq_len,
        augment_on_the_fly=False,
        dataset_split="val",
        deterministic_variants=det_specs,
        augment_seed=augment_seed,
        per_uid_variants=per_uid_map,
    )
    test_ds = PoseDataset(
        test_ex,
        label_to_idx,
        seq_len,
        augment_on_the_fly=False,
        dataset_split="test",
        deterministic_variants=det_specs,
        augment_seed=augment_seed,
        per_uid_variants=per_uid_map,
    )

    def _hard_negative_uid_boost_weights(base_weights: Optional[List[float]] = None) -> Optional[List[float]]:
        if not hard_negative_uid_set:
            return base_weights
        hn = hard_negative_uid_set
        mult = float(max(1.0, hard_negative_uid_weight))
        if base_weights is None:
            weights = [1.0] * len(train_ex)
        else:
            weights = list(base_weights)
        boosted = 0
        for i, ex in enumerate(train_ex):
            uids = _manifest_lookup_uids(ex)
            if any(u in hn for u in uids):
                weights[i] *= mult
                boosted += 1
        print(
            f"[HARD-NEG-UID] sampler boost | uids={len(hn)} filas_boost={boosted} weight×{mult:.2f}"
        )
        return weights

    if balanced and task == "binary":
        # WeightedRandomSampler para reducir desbalance (en binario: labels 0/1)
        train_labels = [ex.label for ex in train_ex]
        count0 = sum(1 for v in train_labels if v == 0)
        count1 = sum(1 for v in train_labels if v == 1)
        if count0 > 0 and count1 > 0:
            class_weights = {0: 1.0 / count0, 1: 1.0 / count1}
            sample_weights = [class_weights[v] for v in train_labels]
            sample_weights = _hard_negative_uid_boost_weights(sample_weights) or sample_weights
            sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=num_workers,
            )
            print(f"[BALANCED-TRAIN] binary sampler activo | count0={count0} count1={count1}")
        else:
            hn_weights = _hard_negative_uid_boost_weights()
            if hn_weights:
                sampler = WeightedRandomSampler(
                    hn_weights, num_samples=len(hn_weights), replacement=True
                )
                train_loader = DataLoader(
                    train_ds,
                    batch_size=batch_size,
                    sampler=sampler,
                    shuffle=False,
                    num_workers=num_workers,
                )
            else:
                train_loader = DataLoader(
                    train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
                )
            print(f"[BALANCED-TRAIN] binary sampler ignorado (count0={count0}, count1={count1})")
    elif maintain_class_ratio and task == "binary":
        train_labels = [ex.label for ex in train_ex]
        count0 = sum(1 for v in train_labels if v == 0)
        count1 = sum(1 for v in train_labels if v == 1)
        if count0 > 0 and count1 > 0:
            ratio = float(target_neg_pos_ratio) if target_neg_pos_ratio is not None else (count0 / count1)
            p_neg = ratio / (1.0 + ratio)
            p_pos = 1.0 / (1.0 + ratio)
            class_weights = {0: p_neg / count0, 1: p_pos / count1}
            sample_weights = [class_weights[v] for v in train_labels]
            sample_weights = _hard_negative_uid_boost_weights(sample_weights) or sample_weights
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=num_workers,
            )
            print(
                "[RATIO-TRAIN] sampler activo | "
                f"count0={count0} count1={count1} target_neg_pos_ratio={ratio:.3f}"
            )
        else:
            hn_weights = _hard_negative_uid_boost_weights()
            if hn_weights:
                sampler = WeightedRandomSampler(
                    hn_weights, num_samples=len(hn_weights), replacement=True
                )
                train_loader = DataLoader(
                    train_ds,
                    batch_size=batch_size,
                    sampler=sampler,
                    shuffle=False,
                    num_workers=num_workers,
                )
            else:
                train_loader = DataLoader(
                    train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
                )
            print(f"[RATIO-TRAIN] sampler ignorado (count0={count0}, count1={count1})")
    else:
        hn_weights = _hard_negative_uid_boost_weights()
        if hn_weights:
            sampler = WeightedRandomSampler(
                hn_weights, num_samples=len(hn_weights), replacement=True
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=num_workers,
            )
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    sample_x, _ = train_ds[0]
    input_dim = sample_x.shape[-1]

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    split_manifest = {
        "split_uids_clips": split_uids_clips,
        "training_plan_path": str(training_plan_path) if training_plan_path else None,
        "split_ratios": {"train": tr, "val": vr, "test": te},
        "augment_profile": augment_profile,
        "deterministic_variants_count": len(det_specs),
        "manifest_cache_dir": str(manifest_cache_dir) if manifest_cache_dir else None,
        "manifest_variant_set": manifest_variant_set,
        "manifest_uids_with_cache": manifest_hits,
        "extra_manifest_views_per_clip": int(extra_manifest_views_per_clip),
        "category_augmentation_config": str(cat_aug_path),
        "category_augmentation_enabled": bool(_category_augment_is_active(cat_aug_cfg)),
        "training_pool_stats": {
            "train_unique_npy": int(train_unique_npy),
            "train_rows": int(len(train_ex)),
            "train_rows_before_expand": int(train_rows_before_expand),
            "train_rows_grid_only": int(tr_g),
            "train_rows_manifest_identity": int(tr_mi),
            "train_rows_manifest_ops": int(tr_mo),
            "val_unique_npy": int(val_unique_npy),
            "val_rows": int(len(val_ex)),
            "test_unique_npy": int(test_unique_npy),
            "test_rows": int(len(test_ex)),
            "deterministic_grid_variant_specs": int(len(det_specs)),
            "manifest_uids_loaded": int(manifest_hits),
        },
        "augment_policy": {
            "train": "category/manifest expand (solo train) + deterministic_grid (opcional) + random on-the-fly (opcional)",
            "val": "sin augmentación (clip original)",
            "test": "sin augmentación (clip original)",
            "note": "Val/test no reciben expansión por categoría ni variantes deterministas/aleatorias.",
        },
    }
    return loaders, input_dim, label_to_idx, split_manifest


def build_model(arch: str, input_dim: int, num_classes: int, cfg: Dict[str, Any]) -> nn.Module:
    if arch == "tcn":
        return PoseTCNClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=cfg.get("hidden_dim", 128),
            dropout=cfg.get("dropout", 0.1),
        )
    if arch == "res_tcn":
        return PoseResTCNClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=cfg.get("hidden_dim", 128),
            num_blocks=cfg.get("num_blocks", 3),
            dropout=cfg.get("dropout", 0.1),
        )
    if arch == "dilated_tcn":
        return PoseDilatedTCNClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=cfg.get("hidden_dim", 128),
            num_layers=cfg.get("num_layers", 4),
            dropout=cfg.get("dropout", 0.1),
        )
    if arch == "stgcn":
        return PoseSTGCNClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=cfg.get("hidden_dim", 128),
            dropout=cfg.get("dropout", 0.1),
        )
    if arch == "lstm":
        return PoseLSTMClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=cfg.get("hidden_dim", 128),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.1),
        )
    if arch == "transformer":
        return PoseTransformerClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=cfg.get("d_model", 128),
            nhead=cfg.get("nhead", 4),
            num_layers=cfg.get("num_layers", 2),
            dim_feedforward=cfg.get("dim_feedforward", 256),
            dropout=cfg.get("dropout", 0.1),
        )
    if arch == "pose_cnn2d":
        return PoseCNN2DClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=cfg.get("hidden_dim", 64),
            dropout=cfg.get("dropout", 0.1),
        )
    if arch == "joint_attn":
        return PoseJointAttnClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            joint_d_model=cfg.get("joint_d_model", 64),
            temporal_d_model=cfg.get("temporal_d_model", 128),
            joint_layers=cfg.get("joint_layers", 1),
            temporal_layers=cfg.get("temporal_layers", 2),
            nhead=cfg.get("nhead", 4),
            dim_feedforward=cfg.get("dim_feedforward", 256),
            dropout=cfg.get("dropout", 0.1),
        )
    if arch == "tcn_lstm":
        return PoseTCNLSTMClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            tcn_hidden_dim=cfg.get("tcn_hidden_dim", 128),
            tcn_layers=cfg.get("tcn_layers", 2),
            lstm_hidden_dim=cfg.get("lstm_hidden_dim", 128),
            lstm_layers=cfg.get("lstm_layers", 1),
            dropout=cfg.get("dropout", 0.1),
        )
    if arch == "gru":
        return PoseGRUClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=cfg.get("hidden_dim", 128),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.1),
        )
    if arch == "gru_attn":
        return PoseGRUAttnClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=cfg.get("hidden_dim", 128),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.1),
        )
    if arch == "tcn_gru":
        return PoseTCNGRUClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            tcn_hidden_dim=cfg.get("tcn_hidden_dim", 128),
            tcn_layers=cfg.get("tcn_layers", 2),
            gru_hidden_dim=cfg.get("gru_hidden_dim", 128),
            gru_layers=cfg.get("gru_layers", 1),
            dropout=cfg.get("dropout", 0.1),
        )
    if arch == "conformer_lite":
        return PoseConformerLiteClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=cfg.get("d_model", 128),
            nhead=cfg.get("nhead", 4),
            num_layers=cfg.get("num_layers", 2),
            dim_feedforward=cfg.get("dim_feedforward", 256),
            dropout=cfg.get("dropout", 0.1),
            conv_kernel=cfg.get("conv_kernel", 7),
        )
    if arch == "ms_tcn":
        return PoseMSTCNClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=cfg.get("hidden_dim", 128),
            num_blocks=cfg.get("num_blocks", 3),
            dropout=cfg.get("dropout", 0.1),
        )
    if arch == "tcn_attn":
        return PoseTCNAttnClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=cfg.get("hidden_dim", 128),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.1),
        )
    if arch == "gat_tcn":
        return PoseGATTCNClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=cfg.get("hidden_dim", 64),
            tcn_hidden_dim=cfg.get("tcn_hidden_dim", 128),
            dropout=cfg.get("dropout", 0.1),
        )
    raise ValueError(f"Arquitectura desconocida: {arch}")


def run_experiment(
    exp_id: int,
    cfg: Dict[str, Any],
    device: torch.device,
    task: str = "multiclass",
    positive_class: int = 6,
    pose_source_override: str | None = None,
    balanced: bool = False,
    single_user_only: bool = False,
    models_dir: Path = MODELS_DIR,
    min_clip_seconds: float = MIN_CLIP_SECONDS,
    min_valid_frames: int = MIN_VALID_FRAMES,
    min_valid_pct: float = MIN_VALID_PCT,
    max_occlusion_ratio: float = MAX_OCCLUSION_RATIO,
    augment_on_the_fly: bool = False,
    augment_config_path: Path = AUGMENT_CONFIG_PATH,
    augment_profile: str = AUGMENT_PROFILE_DEFAULT,
    augment_prob: float = AUGMENT_PROB,
    augment_max_ops: int = AUGMENT_MAX_OPS,
    augment_seed: int = SEED,
    maintain_class_ratio: bool = False,
    target_neg_pos_ratio: Optional[float] = None,
    split_manifest_out: Optional[Path] = None,
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
    asym_gamma_neg: float = 4.0,
    asym_gamma_pos: float = 1.0,
    hard_negative_mining: bool = False,
    hard_negative_topk_frac: float = 0.2,
    hard_negative_weight: float = 1.5,
    mixstyle_prob: float = 0.0,
    mixstyle_alpha: float = 0.3,
    ssl_warmup_epochs: int = 0,
    ssl_consistency_weight: float = 0.0,
    ssl_noise_std: float = 0.01,
    ssl_drop_prob: float = 0.05,
    train_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
    max_deterministic_variants: int = MAX_DETERMINISTIC_VARIANTS,
    train_deterministic_prob: float = TRAIN_DETERMINISTIC_PROB,
    use_deterministic_in_train: bool = True,
    manifest_cache_dir: Optional[Path] = None,
    manifest_variant_set: str = MANIFEST_VARIANT_SET_DEFAULT,
    extra_manifest_views_per_clip: int = 0,
    category_aug_config_path: Optional[Path] = None,
    use_category_augmentation: Optional[bool] = None,
    training_plan_path: Optional[Path] = None,
    stratified_split: bool = True,
    binary_softmax_threshold: float = DEFAULT_BINARY_SOFTMAX_THRESHOLD,
    binary_logit_margin: float = DEFAULT_BINARY_LOGIT_MARGIN,
    hard_negative_uid_manifest: Optional[Path] = None,
    hard_negative_uid_weight: float = 3.0,
) -> Dict[str, Any]:
    t_exp0 = time.perf_counter()
    print("\n" + "=" * 80)
    print(f"Experimento {exp_id:02d} | config={cfg}")
    print("=" * 80)

    seq_len = cfg.get("seq_len", 64)
    batch_size = cfg.get("batch_size", 32)
    lr = cfg.get("lr", 1e-3)
    epochs = cfg.get("epochs", 20)
    pose_source = pose_source_override or cfg.get("pose_source", "filtered")

    hard_negative_uid_set: Optional[set] = None
    hn_weight = float(hard_negative_uid_weight)
    if hard_negative_uid_manifest is not None:
        with open(hard_negative_uid_manifest, "r", encoding="utf-8") as f:
            hn_data = json.load(f)
        hard_negative_uid_set = {str(u) for u in hn_data.get("uids", [])}
        hn_weight = float(hn_data.get("uid_weight", hn_weight))
        print(
            f"[HARD-NEG-UID] manifest={hard_negative_uid_manifest} | "
            f"uids={len(hard_negative_uid_set)} weight={hn_weight:.2f}"
        )

    loaders, input_dim, label_to_idx, split_manifest = build_datasets_and_loaders(
        seq_len=seq_len,
        batch_size=batch_size,
        pose_source=pose_source,
        task=task,
        positive_class=positive_class,
        balanced=balanced,
        single_user_only=single_user_only,
        min_clip_seconds=min_clip_seconds,
        min_valid_frames=min_valid_frames,
        min_valid_pct=min_valid_pct,
        max_occlusion_ratio=max_occlusion_ratio,
        augment_on_the_fly=augment_on_the_fly,
        augment_config_path=augment_config_path,
        augment_profile=augment_profile,
        augment_prob=augment_prob,
        augment_max_ops=augment_max_ops,
        augment_seed=augment_seed,
        maintain_class_ratio=maintain_class_ratio,
        target_neg_pos_ratio=target_neg_pos_ratio,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        max_deterministic_variants=max_deterministic_variants,
        train_deterministic_prob=train_deterministic_prob,
        use_deterministic_in_train=use_deterministic_in_train,
        manifest_cache_dir=manifest_cache_dir,
        manifest_variant_set=manifest_variant_set,
        extra_manifest_views_per_clip=extra_manifest_views_per_clip,
        category_aug_config_path=category_aug_config_path,
        use_category_augmentation=use_category_augmentation,
        training_plan_path=training_plan_path,
        stratified_split=stratified_split,
        hard_negative_uid_set=hard_negative_uid_set,
        hard_negative_uid_weight=hn_weight,
    )
    num_classes = len(label_to_idx)

    eval_kwargs = dict(
        task=task,
        binary_softmax_threshold=binary_softmax_threshold,
        binary_logit_margin=binary_logit_margin,
    )

    model = build_model(cfg["arch"], input_dim, num_classes, cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.get("weight_decay", 0.0))

    best_val_acc = 0.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        in_ssl_warmup = epoch <= int(max(0, ssl_warmup_epochs))
        train_loss = train_one_epoch_advanced(
            model,
            loaders["train"],
            optimizer,
            device,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            asym_gamma_neg=asym_gamma_neg,
            asym_gamma_pos=asym_gamma_pos,
            hard_negative_mining=hard_negative_mining,
            hard_negative_topk_frac=hard_negative_topk_frac,
            hard_negative_weight=hard_negative_weight,
            mixstyle_prob=mixstyle_prob,
            mixstyle_alpha=mixstyle_alpha,
            ssl_consistency_weight=ssl_consistency_weight,
            ssl_noise_std=ssl_noise_std,
            ssl_drop_prob=ssl_drop_prob,
            supervised_weight=(0.0 if in_ssl_warmup else 1.0),
        )
        val_loss, val_acc, val_metrics = evaluate_with_metrics(
            model,
            loaders["val"],
            criterion,
            device,
            num_classes=num_classes,
            **eval_kwargs,
        )
        val_msg = (
            f"[Exp {exp_id:02d}] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )
        if task == "binary" and val_metrics.get("binary"):
            extra = format_binary_metrics_line(val_metrics["binary"])
            if extra:
                val_msg += f" | {extra}"
        elif task == "multiclass" and num_classes > 2:
            per_acc = val_metrics.get("per_class_accuracy_pct", {})
            if per_acc:
                acc_parts = [
                    f"c{k}={v:.0f}%" for k, v in sorted(per_acc.items(), key=lambda x: int(x[0]))
                    if v is not None
                ]
                if acc_parts:
                    val_msg += " | " + " ".join(acc_parts[:8])
                    if len(acc_parts) > 8:
                        val_msg += " ..."
        print(val_msg)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "val_metrics": val_metrics,
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[Exp {exp_id:02d}] Mejor val_acc: {best_val_acc:.4f}")

    test_loss, test_acc, test_metrics = evaluate_with_metrics(
        model,
        loaders["test"],
        criterion,
        device,
        num_classes=num_classes,
        **eval_kwargs,
    )
    # Métricas agregadas en test (clips de un único usuario)
    macro_f1 = test_metrics["macro_f1"]
    weighted_f1 = test_metrics["weighted_f1"]

    # Si estamos en modo binario, extraemos también métricas específicas de la clase positiva (robos)
    f1_pos = None
    rec_pos = None
    prec_pos = None
    if task == "binary":
        # En binario, tras make_binary_examples, la clase 1 es la positiva
        # label_to_idx mapea label_original_binaria -> índice interno (normalmente {0:0, 1:1})
        pos_label = 1
        if pos_label in label_to_idx:
            pos_idx = label_to_idx[pos_label]
            pos_stats = test_metrics["per_class"].get(pos_idx, {})
            prec_pos = float(pos_stats.get("precision", 0.0))
            rec_pos = float(pos_stats.get("recall", 0.0))
            f1_pos = float(pos_stats.get("f1", 0.0))

    # Log a consola
    base_msg = (
        f"[Exp {exp_id:02d}] Test | "
        f"loss={test_loss:.4f} | acc={test_acc:.4f} | "
        f"macro_f1={macro_f1:.4f} | "
        f"weighted_f1={weighted_f1:.4f}"
    )
    if f1_pos is not None:
        base_msg += (
            f" | f1_pos={f1_pos:.4f} | "
            f"rec_pos={rec_pos:.4f} | prec_pos={prec_pos:.4f} "
            f"(clase positiva={positive_class})"
        )
    if task == "binary" and test_metrics.get("binary"):
        extra = format_binary_metrics_line(test_metrics["binary"], prefix="test: ")
        if extra:
            base_msg += f" | {extra}"
    print(base_msg)

    save_path = models_dir / f"modelo_{exp_id:02d}.pt"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "label_to_idx": label_to_idx,
        "config": cfg,
        "input_dim": input_dim,
        "seq_len": seq_len,
        "task": task,
        "positive_class": positive_class,
        "num_classes": num_classes,
        "metrics": {
            "best_val_acc": float(best_val_acc),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_macro_f1": float(macro_f1),
            "test_weighted_f1": float(weighted_f1),
            "test_f1_pos": float(f1_pos) if f1_pos is not None else None,
            "test_rec_pos": float(rec_pos) if rec_pos is not None else None,
            "test_prec_pos": float(prec_pos) if prec_pos is not None else None,
            "test_top3_acc": float(test_metrics["top3_acc"]),
            "test_confusion_matrix": test_metrics["confusion_matrix"],
            "test_per_class": test_metrics["per_class"],
            "test_binary_metrics": test_metrics.get("binary"),
            "test_per_class_accuracy_pct": test_metrics.get("per_class_accuracy_pct"),
            "history": history,
        },
        "split_manifest_path": str(split_manifest_out) if split_manifest_out is not None else None,
        "split_ratios": split_manifest.get("split_ratios"),
        "augment_profile": augment_profile,
        "augment_config_path": str(augment_config_path),
        "deterministic_variants_count": split_manifest.get("deterministic_variants_count"),
        "manifest_cache_dir": split_manifest.get("manifest_cache_dir"),
        "manifest_variant_set": split_manifest.get("manifest_variant_set"),
        "manifest_uids_with_cache": split_manifest.get("manifest_uids_with_cache"),
        "extra_manifest_views_per_clip": split_manifest.get("extra_manifest_views_per_clip"),
        "advanced_training": {
            "loss_type": loss_type,
            "focal_gamma": float(focal_gamma),
            "asym_gamma_neg": float(asym_gamma_neg),
            "asym_gamma_pos": float(asym_gamma_pos),
            "hard_negative_mining": bool(hard_negative_mining),
            "hard_negative_topk_frac": float(hard_negative_topk_frac),
            "hard_negative_weight": float(hard_negative_weight),
            "mixstyle_prob": float(mixstyle_prob),
            "mixstyle_alpha": float(mixstyle_alpha),
            "ssl_warmup_epochs": int(ssl_warmup_epochs),
            "ssl_consistency_weight": float(ssl_consistency_weight),
            "ssl_noise_std": float(ssl_noise_std),
            "ssl_drop_prob": float(ssl_drop_prob),
        },
    }

    torch.save(checkpoint, save_path)
    print(f"[Exp {exp_id:02d}] Modelo guardado en: {save_path}")
    if split_manifest_out is not None:
        split_payload = {
            "version": 1,
            "seed": SEED,
            "task": task,
            "positive_class": int(positive_class),
            "pose_source": str(pose_source),
            "single_user_only": bool(single_user_only),
            "filters": {
                "min_clip_seconds": float(min_clip_seconds),
                "min_valid_frames": int(min_valid_frames),
                "min_valid_pct": float(min_valid_pct),
                "max_occlusion_ratio": float(max_occlusion_ratio),
            },
            "split": {
                "split_uids_clips": split_manifest["split_uids_clips"],
            },
            "split_ratios": split_manifest.get("split_ratios"),
            "augment_profile": split_manifest.get("augment_profile"),
            "deterministic_variants_count": split_manifest.get("deterministic_variants_count"),
            "augment_policy": split_manifest.get("augment_policy"),
            "extra_manifest_views_per_clip": split_manifest.get("extra_manifest_views_per_clip"),
        }
        with open(split_manifest_out, "w", encoding="utf-8") as f:
            json.dump(split_payload, f, indent=2, ensure_ascii=False)
        print(f"[Exp {exp_id:02d}] Split manifest guardado en: {split_manifest_out}")

    wall_s = float(time.perf_counter() - t_exp0)
    print(
        f"[Exp {exp_id:02d}] Tiempo total del experimento: {wall_s:.1f} s "
        f"({wall_s / 60.0:.2f} min)"
    )

    return {
        "exp_id": exp_id,
        "config": cfg,
        "wall_time_s": wall_s,
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_macro_f1": float(macro_f1),
        "test_weighted_f1": float(weighted_f1),
        "test_f1_pos": float(f1_pos) if f1_pos is not None else None,
        "test_rec_pos": float(rec_pos) if rec_pos is not None else None,
        "test_prec_pos": float(prec_pos) if prec_pos is not None else None,
        "save_path": str(save_path),
        "split_manifest_path": (str(split_manifest_out) if split_manifest_out is not None else None),
    }


def _select_debug_experiments(experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    En modo debug seleccionamos:
      - El experimento más simple (menos epochs; si empatan, el primero) de cada arquitectura.
    """
    by_arch: Dict[str, Dict[str, Any]] = {}
    for cfg in experiments:
        arch = cfg.get("arch")
        if arch is None:
            continue
        if cfg.get("done", False):
            continue
        epochs = int(cfg.get("epochs", 0))
        if arch not in by_arch or epochs < int(by_arch[arch].get("epochs", 1e9)):
            by_arch[arch] = cfg
    selected = [by_arch[a] for a in sorted(by_arch.keys())]
    print(f"[DEBUG] Experimentos seleccionados (uno por arquitectura):")
    for cfg in selected:
        print(f"  - arch={cfg['arch']}, epochs={cfg.get('epochs')}, batch={cfg.get('batch_size')}, seq_len={cfg.get('seq_len')}")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos de acción sobre poses.")
    parser.add_argument(
        "--task",
        choices=["multiclass", "binary"],
        default="multiclass",
        help="Tipo de tarea: 'multiclass' (por defecto) o 'binary' (robo vs no-robo).",
    )
    parser.add_argument(
        "--positive-class",
        type=int,
        default=6,
        help="Etiqueta original considerada positiva en modo binario (por defecto 6 = robos).",
    )
    parser.add_argument(
        "--pose-source",
        choices=["filtered", "full"],
        default=None,
        help="Sobrescribe pose_source de los experimentos: 'filtered' (poses.npy) o 'full' (poses_full.npy).",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Balancea el muestreo en modo binario para reducir el desbalance (WeightedRandomSampler).",
    )
    parser.add_argument(
        "--single-user-only",
        action="store_true",
        help=(
            "Si se indica: solo clips con exactamente un usuario. "
            "Por defecto (sin flag): modo multiusuario — todos los usuarios válidos por clip."
        ),
    )
    parser.add_argument(
        "--min-clip-seconds",
        type=float,
        default=MIN_CLIP_SECONDS,
        help=f"Duración mínima del clip en segundos (default {MIN_CLIP_SECONDS}).",
    )
    parser.add_argument(
        "--min-valid-frames",
        type=int,
        default=MIN_VALID_FRAMES,
        help=f"Frames válidos mínimos por usuario (default {MIN_VALID_FRAMES}).",
    )
    parser.add_argument(
        "--min-valid-pct",
        type=float,
        default=MIN_VALID_PCT,
        help=f"Porcentaje mínimo de frames válidos por usuario (default {MIN_VALID_PCT}).",
    )
    parser.add_argument(
        "--max-occlusion-ratio",
        type=float,
        default=MAX_OCCLUSION_RATIO,
        help=f"Occlusión máxima permitida por usuario (default {MAX_OCCLUSION_RATIO}).",
    )
    parser.add_argument("--augment-on-the-fly", action="store_true", help="Aplica augment en memoria durante training.")
    parser.add_argument(
        "--augment-config",
        type=str,
        default=str(AUGMENT_CONFIG_PATH),
        help="Ruta a validate_npy.json para obtener rangos por perfil.",
    )
    parser.add_argument(
        "--augment-profile",
        type=str,
        default=AUGMENT_PROFILE_DEFAULT,
        help=f"Perfil de augment en validate_npy.json (default {AUGMENT_PROFILE_DEFAULT}).",
    )
    parser.add_argument(
        "--augment-prob",
        type=float,
        default=AUGMENT_PROB,
        help=f"Probabilidad de aplicar augment por muestra (default {AUGMENT_PROB}).",
    )
    parser.add_argument(
        "--augment-max-ops",
        type=int,
        default=AUGMENT_MAX_OPS,
        help=f"Máximo de operaciones por muestra aumentada (default {AUGMENT_MAX_OPS}).",
    )
    parser.add_argument("--augment-seed", type=int, default=SEED, help="Semilla de augment on-the-fly.")
    parser.add_argument(
        "--maintain-class-ratio",
        action="store_true",
        help="En binario, usa sampler para mantener ratio objetivo no-robo/robo tras augment.",
    )
    parser.add_argument(
        "--target-neg-pos-ratio",
        type=float,
        default=None,
        help="Ratio objetivo no-robo/robo para sampler en binario. Si no se indica, usa ratio observado.",
    )
    parser.add_argument("--loss-type", choices=["ce", "focal", "asymmetric"], default="ce", help="Pérdida de entrenamiento.")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Gamma para focal loss.")
    parser.add_argument("--asym-gamma-neg", type=float, default=4.0, help="Gamma negativo para asymmetric loss.")
    parser.add_argument("--asym-gamma-pos", type=float, default=1.0, help="Gamma positivo para asymmetric loss.")
    parser.add_argument("--hard-negative-mining", action="store_true", help="Repondera negativos más confusos (binario).")
    parser.add_argument("--hard-negative-topk-frac", type=float, default=0.2, help="Fracción de negativos duros por batch.")
    parser.add_argument("--hard-negative-weight", type=float, default=1.5, help="Peso extra para negativos duros.")
    parser.add_argument("--mixstyle-prob", type=float, default=0.0, help="Probabilidad MixStyle para robustez de dominio.")
    parser.add_argument("--mixstyle-alpha", type=float, default=0.3, help="Alpha de beta en MixStyle.")
    parser.add_argument("--ssl-warmup-epochs", type=int, default=0, help="Épocas iniciales solo consistencia (sin CE).")
    parser.add_argument("--ssl-consistency-weight", type=float, default=0.0, help="Peso de consistencia SSL en train.")
    parser.add_argument("--ssl-noise-std", type=float, default=0.01, help="Ruido en vista SSL.")
    parser.add_argument("--ssl-drop-prob", type=float, default=0.05, help="Drop de features en vista SSL.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help=(
            "Fracción train. Si omites --train-ratio y --val-ratio, se usa suggest_split_ratios(N) "
            f"(misma heurística que preflight). Referencia en model_config: {SPLIT_RATIO_TRAIN:.3f}."
        ),
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help=(
            "Fracción val (test = 1 - train - val). Omitir junto con --train-ratio para heurística automática."
        ),
    )
    parser.add_argument(
        "--max-deterministic-variants",
        type=int,
        default=MAX_DETERMINISTIC_VARIANTS,
        help=f"Tope de variantes deterministas (rejilla validate_npy) por perfil (default {MAX_DETERMINISTIC_VARIANTS}).",
    )
    parser.add_argument(
        "--train-deterministic-prob",
        type=float,
        default=TRAIN_DETERMINISTIC_PROB,
        help=(
            "En train, prob. de aplicar una variante determinista antes del augment aleatorio "
            f"(default {TRAIN_DETERMINISTIC_PROB})."
        ),
    )
    parser.add_argument(
        "--no-train-deterministic",
        action="store_true",
        help="En train, no aplicar variantes deterministas (solo identidad + augment aleatorio si está activo).",
    )
    parser.add_argument(
        "--manifest-cache-dir",
        type=str,
        default=None,
        help=(
            "Directorio con un JSON por UID (md5 de la ruta absoluta del .npy) generado por validate_npy. "
            "Por defecto no se usa; entrenamiento usa solo rejilla global. "
            f"Valor típico: {MANIFEST_CACHE_DIR}"
        ),
    )
    parser.add_argument(
        "--manifest-variant-set",
        type=str,
        choices=["min", "industrial", "full"],
        default=MANIFEST_VARIANT_SET_DEFAULT,
        help=f"Qué lista del manifest validate_npy usar por UID (selected_n_*). Default {MANIFEST_VARIANT_SET_DEFAULT}.",
    )
    parser.add_argument(
        "--extra-manifest-views-per-clip",
        type=int,
        default=EXTRA_MANIFEST_VIEWS_PER_CLIP_DEFAULT,
        help=(
            "Expande train/val: por clip con manifest, 1 fila identidad + hasta N variantes validate_npy "
            "(sin escribir .npy). Test sin expandir. Requiere --manifest-cache-dir. "
            f"Default {EXTRA_MANIFEST_VIEWS_PER_CLIP_DEFAULT}."
        ),
    )
    parser.add_argument(
        "--category-aug-config",
        type=str,
        default=None,
        help=(
            "JSON augment por categoría. Default según --task "
            "(config_category_augmentation.json vs _binary.json)."
        ),
    )
    parser.add_argument(
        "--no-category-augmentation",
        action="store_true",
        help="Desactiva la expansión por categoría aunque el JSON tenga counts > 0.",
    )
    parser.add_argument(
        "--training-plan",
        type=str,
        default=None,
        help=(
            "Plan JSON (default: training_plan.json o training_plan_binary.json según --task)."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Carpeta de salida modelo_*.pt (default separada por task binario/multiclase).",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default=None,
        help="Carpeta split_manifest_exp_*.json (default splits/ o splits_binary/).",
    )
    parser.add_argument(
        "--no-stratified-split",
        action="store_true",
        help="Sin training_plan: usa split aleatorio global en lugar de estratificado por UID/categoría.",
    )
    parser.add_argument(
        "--binary-softmax-threshold",
        type=float,
        default=DEFAULT_BINARY_SOFTMAX_THRESHOLD,
        help=f"Umbral P(robo) en eval binaria (default {DEFAULT_BINARY_SOFTMAX_THRESHOLD}).",
    )
    parser.add_argument(
        "--binary-logit-margin",
        type=float,
        default=DEFAULT_BINARY_LOGIT_MARGIN,
        help=(
            "Margen logit[1]-logit[0] para predicción binaria alternativa "
            f"(default {DEFAULT_BINARY_LOGIT_MARGIN})."
        ),
    )
    return parser.parse_args()


def main():
    # Redirección de logs: terminal + fichero en training/logs/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"train_{timestamp}.log"

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                    if hasattr(s, "flush"):
                        s.flush()
                except Exception:
                    pass

        def flush(self):
            for s in self.streams:
                try:
                    if hasattr(s, "flush"):
                        s.flush()
                except Exception:
                    pass

    original_stdout = sys.stdout
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(original_stdout, log_file)

    try:
        reset_training_pool_summary_flag()
        args = parse_args()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("\n" + "=" * 80)
        if args.single_user_only:
            print("MODO DE DATOS: single-user (solo clips con exactamente un usuario)")
        else:
            print("MODO DE DATOS: MULTIUSUARIO por defecto (todos los usuarios válidos por clip)")
        print("=" * 80 + "\n")
        print(f"Usando device: {device}")

        artifacts = resolve_artifacts(
            args.task,
            single_user_only=args.single_user_only,
            training_plan=args.training_plan,
            category_aug_config=args.category_aug_config,
            models_dir=args.models_dir,
            splits_dir=args.splits_dir,
        )
        print_artifact_banner(artifacts, title=f"Entrenamiento ({args.task})")

        models_dir = artifacts["models_dir"]
        splits_dir = artifacts["splits_dir"]
        category_aug_path = artifacts["category_aug_config"]
        if args.training_plan:
            training_plan_path: Optional[Path] = Path(args.training_plan)
        elif artifacts["training_plan"].is_file():
            training_plan_path = artifacts["training_plan"]
        else:
            training_plan_path = None

        print(f"Log de esta sesión: {log_path}")
        print(f"Tarea: {args.task} | positive_class={args.positive_class} | pose_source_override={args.pose_source}")
        print(f"single_user_only={args.single_user_only}")
        print(
            "Filtros de calidad => "
            f"min_clip_seconds={args.min_clip_seconds}, "
            f"min_valid_frames={args.min_valid_frames}, "
            f"min_valid_pct={args.min_valid_pct}, "
            f"max_occlusion_ratio={args.max_occlusion_ratio}"
        )
        print(
            "Augment => "
            f"on_the_fly={args.augment_on_the_fly}, profile={args.augment_profile}, "
            f"prob={args.augment_prob}, max_ops={args.augment_max_ops}, "
            f"maintain_class_ratio={args.maintain_class_ratio}, target_neg_pos_ratio={args.target_neg_pos_ratio}"
        )
        print(
            "Avanzado => "
            f"loss={args.loss_type}, hard_neg={args.hard_negative_mining}, "
            f"mixstyle_p={args.mixstyle_prob}, ssl_warmup={args.ssl_warmup_epochs}, ssl_w={args.ssl_consistency_weight}"
        )
        split_cli = (
            f"train_ratio={args.train_ratio}, val_ratio={args.val_ratio}"
            if args.train_ratio is not None and args.val_ratio is not None
            else "train/val = heurística suggest_split_ratios(N) (omitidos en CLI)"
        )
        print(
            "Split / augment determinista => "
            f"{split_cli}, "
            f"max_det_variants={args.max_deterministic_variants}, "
            f"train_det_prob={args.train_deterministic_prob}, "
            f"use_train_det={not args.no_train_deterministic}"
        )
        mcache = Path(args.manifest_cache_dir) if args.manifest_cache_dir else None
        print(
            "Manifest cache => "
            f"dir={mcache} | variant_set={args.manifest_variant_set}"
            + (" (activo)" if mcache else " (desactivado)")
        )
        print(
            f"Extra manifest views / clip (train+val) => {args.extra_manifest_views_per_clip} "
            "(0 = sin expansión explícita por filas)"
        )
        bs_thr = float(args.binary_softmax_threshold)
        bs_margin = float(args.binary_logit_margin)
        if training_plan_path is not None and Path(training_plan_path).is_file():
            plan_eval = load_training_plan_json(training_plan_path).get("evaluation", {})
            if bs_thr == DEFAULT_BINARY_SOFTMAX_THRESHOLD:
                bs_thr = float(plan_eval.get("binary_softmax_threshold", bs_thr))
            if bs_margin == DEFAULT_BINARY_LOGIT_MARGIN:
                bs_margin = float(plan_eval.get("binary_logit_margin", bs_margin))
        print(
            f"Eval binaria => softmax_thr={bs_thr}, logit_margin={bs_margin}"
        )

        results = []
        wall_total_s = 0.0
        exps_iter = _select_debug_experiments(EXPERIMENTS) if DEBUG_MODE else EXPERIMENTS
        for i, cfg in enumerate(exps_iter, start=1):
            if (not DEBUG_MODE) and cfg.get("done", False):
                print(f"[Exp {i:02d}] Marcado como done=True, se omite.")
                continue
            res = run_experiment(
                i,
                cfg,
                device,
                task=args.task,
                positive_class=args.positive_class,
                pose_source_override=args.pose_source,
                balanced=args.balanced,
                single_user_only=args.single_user_only,
                models_dir=models_dir,
                min_clip_seconds=args.min_clip_seconds,
                min_valid_frames=args.min_valid_frames,
                min_valid_pct=args.min_valid_pct,
                max_occlusion_ratio=args.max_occlusion_ratio,
                augment_on_the_fly=args.augment_on_the_fly,
                augment_config_path=Path(args.augment_config),
                augment_profile=args.augment_profile,
                augment_prob=args.augment_prob,
                augment_max_ops=args.augment_max_ops,
                augment_seed=args.augment_seed,
                maintain_class_ratio=args.maintain_class_ratio,
                target_neg_pos_ratio=args.target_neg_pos_ratio,
                split_manifest_out=(splits_dir / f"split_manifest_exp_{i:02d}.json"),
                loss_type=args.loss_type,
                focal_gamma=args.focal_gamma,
                asym_gamma_neg=args.asym_gamma_neg,
                asym_gamma_pos=args.asym_gamma_pos,
                hard_negative_mining=args.hard_negative_mining,
                hard_negative_topk_frac=args.hard_negative_topk_frac,
                hard_negative_weight=args.hard_negative_weight,
                mixstyle_prob=args.mixstyle_prob,
                mixstyle_alpha=args.mixstyle_alpha,
                ssl_warmup_epochs=args.ssl_warmup_epochs,
                ssl_consistency_weight=args.ssl_consistency_weight,
                ssl_noise_std=args.ssl_noise_std,
                ssl_drop_prob=args.ssl_drop_prob,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                max_deterministic_variants=args.max_deterministic_variants,
                train_deterministic_prob=args.train_deterministic_prob,
                use_deterministic_in_train=(not args.no_train_deterministic),
                manifest_cache_dir=mcache,
                manifest_variant_set=args.manifest_variant_set,
                extra_manifest_views_per_clip=int(args.extra_manifest_views_per_clip),
                category_aug_config_path=category_aug_path,
                use_category_augmentation=(False if args.no_category_augmentation else None),
                training_plan_path=training_plan_path,
                stratified_split=(not args.no_stratified_split),
                binary_softmax_threshold=bs_thr,
                binary_logit_margin=bs_margin,
            )
            results.append(res)
            wall_total_s += float(res.get("wall_time_s", 0.0))

        print("\n" + "=" * 80)
        print(
            f"Tiempo total acumulado (experimentos ejecutados en esta sesión): "
            f"{wall_total_s:.1f} s ({wall_total_s / 60.0:.2f} min)"
        )
        print("=" * 80 + "\n")

        # Guardar resumen de todos los experimentos (junto al script de training)
        summary_path = BASE_DIR / "experiments_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Resumen de experimentos guardado en: {summary_path}")
    finally:
        # Restaurar stdout y cerrar log
        sys.stdout = original_stdout
        log_file.close()


if __name__ == "__main__":
    main()

