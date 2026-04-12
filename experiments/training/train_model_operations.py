from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
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
        MAX_DETERMINISTIC_VARIANTS,
        TRAIN_DETERMINISTIC_PROB,
        AUGMENT_PROFILE_DEFAULT,
        MANIFEST_VARIANT_SET_DEFAULT,
        EXTRA_MANIFEST_VIEWS_PER_CLIP_DEFAULT,
    )
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
        MAX_DETERMINISTIC_VARIANTS,
        TRAIN_DETERMINISTIC_PROB,
        AUGMENT_PROFILE_DEFAULT,
        MANIFEST_VARIANT_SET_DEFAULT,
        EXTRA_MANIFEST_VIEWS_PER_CLIP_DEFAULT,
    )


# Semillas y splits (por defecto desde model_config; CLI puede sobrescribir en build_datasets)
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
        p = manifest_cache_path_for_uid(manifest_cache_dir, uid)
        if not p.exists():
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if verify_source_path:
                src = data.get("source_npy")
                if src and Path(str(src)).resolve() != Path(uid).resolve():
                    continue
            specs = specs_from_validate_manifest_payload(data, variant_set=variant_set)
            if specs:
                per_uid[uid] = specs
        except Exception:
            continue
    return per_uid


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
    out: List[PoseExample] = []
    skipped = 0
    for ex in examples:
        p = manifest_cache_path_for_uid(mdir, _example_uid(ex))
        if not p.exists():
            out.append(ex)
            skipped += 1
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            out.append(ex)
            skipped += 1
            continue
        specs = specs_from_validate_manifest_payload(data, variant_set=variant_set)
        if not specs:
            out.append(ex)
            skipped += 1
            continue
        out.append(
            PoseExample(
                pose_path=ex.pose_path,
                label=ex.label,
                track_id=ex.track_id,
                clip_name=ex.clip_name,
                category_str=ex.category_str,
                valid_mask_path=ex.valid_mask_path,
                users_in_clip=ex.users_in_clip,
                forced_ops=[],
            )
        )
        n_extra = min(extra_views_per_clip, max(0, len(specs) - 1))
        for j in range(n_extra):
            spec = specs[1 + j]
            ops = list(spec.get("ops", []) or [])
            out.append(
                PoseExample(
                    pose_path=ex.pose_path,
                    label=ex.label,
                    track_id=ex.track_id,
                    clip_name=ex.clip_name,
                    category_str=ex.category_str,
                    valid_mask_path=ex.valid_mask_path,
                    users_in_clip=ex.users_in_clip,
                    forced_ops=ops,
                )
            )
    print(
        f"[EXPAND-VIEWS] extra_manifest_views={extra_views_per_clip} variant_set={variant_set} | "
        f"clips sin manifest (1 fila): {skipped} | filas totales: {len(out)} (antes {len(examples)})"
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
    # Si no es None: aplicar solo estas ops deterministas (manifest validate_npy); train puede añadir on-the-fly después.
    forced_ops: Optional[List[Dict[str, Any]]] = None


def _example_uid(ex: PoseExample) -> str:
    # UID estable para compartir split entre train/eval.
    return str(ex.pose_path.resolve())


def get_data_result_root() -> Path:
    root = DATA_RESULT_ROOT
    if not root.exists():
        raise RuntimeError(f"No se encontró la carpeta data_result en: {root}")
    return root


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
    if clip_duration > 0 and clip_duration < min_clip_seconds:
        return False

    if pose_len < MIN_SEQ_LEN:
        return False
    if valid_frames < min_valid_frames:
        return False
    if valid_pct < min_valid_pct:
        return False
    if occlusion_ratio > max_occlusion_ratio:
        return False

    # Si ya existe un filtro de extracción previo y falla, descartamos.
    passes_filters = user_meta.get("passes_filters")
    if passes_filters is False:
        return False

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
    ops = ["mirror", "rotate", "scale", "shift", "noise"]
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
    np.clip(out, 0.0, 1.0, out=out)
    return out


def collect_examples(
    pose_source: str = "filtered",
    single_user_only: bool = False,
    min_clip_seconds: float = MIN_CLIP_SECONDS,
    min_valid_frames: int = MIN_VALID_FRAMES,
    min_valid_pct: float = MIN_VALID_PCT,
    max_occlusion_ratio: float = MAX_OCCLUSION_RATIO,
) -> List[PoseExample]:
    """
    Recorre data_result/{cat}/{clip_name}/ y construye ejemplos por usuario:
      1) Incluye usuarios tanto en clips de 1 persona como multiusuario (por defecto).
      2) Si single_user_only=True: solo clips con exactamente 1 usuario.
      3) En cat=6 respeta user_cat por usuario:
         - user_cat=6 se mantiene como robo
         - user_cat!=6 se reetiqueta a su user_cat.
      4) En no-cat6 también prioriza user_cat si existe, para etiqueta por usuario.
      5) Aplica filtro de calidad por usuario para descartar secuencias pobres.
      6) pose_source: "filtered" usa poses.npy, "full" usa poses_full.npy (+ valid_mask).
    """
    root = get_data_result_root()
    examples: List[PoseExample] = []

    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat_str = cat_dir.name
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
            "mirror": 0.30,
            "rotate": 0.28,
            "scale": 0.16,
            "shift": 0.16,
            "noise": 0.10,
        }
        self.augment_ranges = augment_ranges or {
            "rotate_degrees": (-15.0, 15.0),
            "scale_percentage": (95.0, 110.0),
            "shift_dx": (-0.05, 0.05),
            "shift_dy": (-0.05, 0.05),
            "noise_sigma_x": (0.0, 0.006),
            "noise_sigma_y": (0.0, 0.006),
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
            if self.dataset_split == "train" and self.augment_on_the_fly:
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
            vidx = _stable_uid_hash(uid) % max(1, n_var)
            spec = variants[vidx]
            ops = spec.get("ops", [])
            det_seed = (_stable_uid_hash(uid) * 1009 + vidx * 17 + _stable_str_hash(str(spec.get("name", "")))) & 0xFFFFFFFF
            rng_det = np.random.default_rng(int(det_seed))
            poses = _apply_deterministic_ops(poses, ops, rng=rng_det)
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
def evaluate_with_metrics(
    model,
    loader,
    criterion,
    device,
    num_classes: int,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Evalúa en un loader y devuelve:
      - pérdida media
      - accuracy
      - métricas detalladas: matriz de confusión, precision/recall/F1 por clase, macro/weighted F1 y top-3 accuracy.
    """
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    conf_mat = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    top3_correct = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        total += x.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()

        # Matriz de confusión
        for yt, yp in zip(y.tolist(), preds.tolist()):
            if 0 <= yt < num_classes and 0 <= yp < num_classes:
                conf_mat[yt][yp] += 1

        # Top-3 accuracy
        if logits.size(1) >= 3:
            top3 = logits.topk(3, dim=1).indices  # [B, 3]
            for yt, topk in zip(y.tolist(), top3.tolist()):
                if yt in topk:
                    top3_correct += 1

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)

    # Métricas derivadas de la matriz de confusión
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

    metrics = {
        "confusion_matrix": conf_mat,
        "per_class": per_class,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "top3_acc": float(top3_acc),
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

    if DEBUG_MODE:
        # Reducir drásticamente el número de ejemplos para pruebas locales rápidas
        examples = examples[:DEBUG_MAX_EXAMPLES]
        print(f"[DEBUG] Usando solo {len(examples)} ejemplos para train/val/test")

    # En modo binario reetiquetamos a 0/1 manteniendo el resto del flujo igual
    if task == "binary":
        print(f"[BINARIO] Usando clase positiva original: {positive_class}")
        examples = make_binary_examples(examples, positive_class=positive_class)

    n_ex = len(examples)
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

    train_ex, val_ex, test_ex = split_examples(examples, seed=SEED, train_ratio=tr, val_ratio=vr)
    print(
        f"Split aplicado => Train: {len(train_ex)} | Val: {len(val_ex)} | Test: {len(test_ex)}"
    )

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
        val_ex = expand_examples_with_manifest_extra_views(
            val_ex,
            Path(manifest_cache_dir),
            manifest_variant_set,
            extra_manifest_views_per_clip,
        )
        print(
            f"Tras expansión manifest => Train filas: {len(train_ex)} | Val filas: {len(val_ex)} | "
            f"Test (sin expandir): {len(test_ex)}"
        )

    label_to_idx = build_label_mapping(examples)
    num_classes = len(label_to_idx)
    print(f"Número de clases: {num_classes} | mapping: {label_to_idx}")

    aug_cfg = _load_augment_profile(augment_config_path, augment_profile)
    if not aug_cfg:
        aug_cfg = {
            "steps": {"rotate": 2.0, "scale": 2.0, "shift": 0.02},
            "rotate": {"min": -15.0, "max": 15.0},
            "noise": {"sigma_cap": 0.006},
        }
    rotate_cfg = aug_cfg.get("rotate", {}) if isinstance(aug_cfg, dict) else {}
    noise_cfg = aug_cfg.get("noise", {}) if isinstance(aug_cfg, dict) else {}
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
    aug_ranges = {
        "rotate_degrees": (float(rotate_cfg.get("min", -15.0)), float(rotate_cfg.get("max", 15.0))),
        "scale_percentage": (95.0, 111.0),
        "shift_dx": (-0.06, 0.06),
        "shift_dy": (-0.06, 0.06),
        "noise_sigma_x": (0.0, float(noise_cfg.get("sigma_cap", 0.006))),
        "noise_sigma_y": (0.0, float(noise_cfg.get("sigma_cap", 0.006))),
    }
    train_ds = PoseDataset(
        train_ex,
        label_to_idx,
        seq_len,
        augment_on_the_fly=augment_on_the_fly,
        augment_prob=augment_prob,
        augment_max_ops=augment_max_ops,
        augment_op_probs={
            "mirror": 0.30,
            "rotate": 0.28,
            "scale": 0.16,
            "shift": 0.16,
            "noise": 0.10,
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

    if balanced and task == "binary":
        # WeightedRandomSampler para reducir desbalance (en binario: labels 0/1)
        train_labels = [ex.label for ex in train_ex]
        count0 = sum(1 for v in train_labels if v == 0)
        count1 = sum(1 for v in train_labels if v == 1)
        if count0 > 0 and count1 > 0:
            class_weights = {0: 1.0 / count0, 1: 1.0 / count1}
            sample_weights = [class_weights[v] for v in train_labels]
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
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
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
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            print(f"[RATIO-TRAIN] sampler ignorado (count0={count0}, count1={count1})")
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    sample_x, _ = train_ds[0]
    input_dim = sample_x.shape[-1]

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    split_manifest = {
        "train": [_example_uid(ex) for ex in train_ex],
        "val": [_example_uid(ex) for ex in val_ex],
        "test": [_example_uid(ex) for ex in test_ex],
        "split_ratios": {"train": tr, "val": vr, "test": te},
        "augment_profile": augment_profile,
        "deterministic_variants_count": len(det_specs),
        "manifest_cache_dir": str(manifest_cache_dir) if manifest_cache_dir else None,
        "manifest_variant_set": manifest_variant_set,
        "manifest_uids_with_cache": manifest_hits,
        "extra_manifest_views_per_clip": int(extra_manifest_views_per_clip),
        "augment_policy": {
            "train": "deterministic_grid (opcional) + random on-the-fly (opcional)",
            "val": "solo determinista (rejilla por UID estable)",
            "test": "solo determinista (rejilla por UID estable)",
            "note": "Val/test no usan muestreo aleatorio en rangos; si hay manifest_cache por UID, se usan variantes validate_npy; si no, rejilla global.",
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
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"Experimento {exp_id:02d} | config={cfg}")
    print("=" * 80)

    seq_len = cfg.get("seq_len", 64)
    batch_size = cfg.get("batch_size", 32)
    lr = cfg.get("lr", 1e-3)
    epochs = cfg.get("epochs", 20)
    pose_source = pose_source_override or cfg.get("pose_source", "filtered")

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
    )
    num_classes = len(label_to_idx)

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
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)
        print(
            f"[Exp {exp_id:02d}] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
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
            f"(clips de un único usuario, clase positiva={positive_class})"
        )
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
                "train": split_manifest["train"],
                "val": split_manifest["val"],
                "test": split_manifest["test"],
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

    return {
        "exp_id": exp_id,
        "config": cfg,
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
        help="Usa solo clips con exactamente un usuario para train/val/test (incluye categoría 6).",
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
        args = parse_args()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando device: {device}")
        models_dir = MODELS_SINGLE_DIR if args.single_user_only else MODELS_DIR
        models_dir.mkdir(parents=True, exist_ok=True)
        print(f"Modelos se guardarán en: {models_dir}")
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

        results = []
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
                split_manifest_out=(SPLITS_DIR / f"split_manifest_exp_{i:02d}.json"),
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
            )
            results.append(res)

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

