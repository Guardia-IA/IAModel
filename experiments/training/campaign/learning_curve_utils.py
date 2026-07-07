"""Utilidades para curva de aprendizaje (train creciente, val/test fijos)."""
from __future__ import annotations

import copy
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from campaign_paths import (
    artifacts_root,
    category_aug_path,
    ensure_cell_dirs,
    training_plan_path,
)

try:
    from preflight_train_plan import write_training_plan
    from train_model_operations import (
        SEED,
        collect_examples,
        count_examples_by_folder_category,
        make_binary_examples,
        propose_category_augment_counts,
        split_examples_by_uid_manifest,
        split_uids_from_example_lists,
        summarize_category_aug_on_train,
        analyze_robbery_augment_balance,
        build_plan_stats_by_category,
        load_category_augmentation_config,
        max_category_augment_ops_available,
        _example_folder_category,
        _example_uid,
    )
    from model_config import ROBBERY_CLASS
except ImportError:
    from ..preflight_train_plan import write_training_plan  # type: ignore
    from ..train_model_operations import (  # type: ignore
        SEED,
        collect_examples,
        count_examples_by_folder_category,
        make_binary_examples,
        propose_category_augment_counts,
        split_examples_by_uid_manifest,
        split_uids_from_example_lists,
        summarize_category_aug_on_train,
        analyze_robbery_augment_balance,
        build_plan_stats_by_category,
        load_category_augmentation_config,
        max_category_augment_ops_available,
        _example_folder_category,
        _example_uid,
    )
    from ..model_config import ROBBERY_CLASS  # type: ignore

LEARNING_CURVE_MASTER_RUN_ID = "_lc_master"
MANIFEST_FILENAME = "learning_curve_manifest.json"
MAX_SIZE_ALIASES = frozenset({"max", "all", "full"})

# bin full/filtered (hardened) + mc full/filtered
DEFAULT_LEARNING_CURVE_CELL_IDS = (
    "mc_full",
    "mc_filtered",
    "bin_full_hardened",
    "bin_filtered_hardened",
)


def learning_curve_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return dict(config.get("learning_curve") or {})


def _is_max_token(value: Any) -> bool:
    return str(value).strip().lower() in MAX_SIZE_ALIASES


def parse_train_size_specs(
    sizes: Optional[Sequence[Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """Devuelve lista de enteros o el token 'max' (todos los clips de train del split maestro)."""
    if sizes is not None:
        raw = list(sizes)
    else:
        lc = learning_curve_config(config or {})
        raw = list(lc.get("train_sizes") or [])
    if not raw:
        raise ValueError(
            "Indica --train-sizes (p. ej. 3000 6500 max) o define learning_curve.train_sizes en campaign_config.json"
        )
    out: List[Any] = []
    for item in raw:
        if _is_max_token(item):
            out.append("max")
            continue
        try:
            n = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Tamaño train inválido: {item!r}. Usa un entero positivo o 'max' (todos los clips train)."
            ) from exc
        if n <= 0:
            raise ValueError(f"Tamaño train debe ser > 0 (got {n})")
        out.append(n)
    return out


def master_train_clip_count(
    cell_id: str,
    master_run_id: str = LEARNING_CURVE_MASTER_RUN_ID,
) -> Optional[int]:
    plan_path = training_plan_path(cell_id, run_id=master_run_id)
    if not plan_path.is_file():
        return None
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    return len((plan.get("split_uids") or {}).get("train") or [])


def load_learning_curve_manifest(
    master_run_id: str = LEARNING_CURVE_MASTER_RUN_ID,
) -> Optional[Dict[str, Any]]:
    path = learning_curve_manifest_path(master_run_id)
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_train_size_specs(
    specs: Sequence[Any],
    *,
    max_train_clips: Optional[int],
) -> List[int]:
    """Resuelve 'max' → número real de clips train del split maestro."""
    if any(_is_max_token(s) for s in specs) and max_train_clips is None:
        raise ValueError(
            "Token 'max' en train_sizes pero no hay split maestro. "
            "Ejecuta preflight --learning-curve --write-all primero."
        )
    resolved: List[int] = []
    cap = int(max_train_clips) if max_train_clips is not None else None
    for spec in specs:
        if _is_max_token(spec):
            resolved.append(cap)  # type: ignore[arg-type]
            continue
        n = int(spec)
        if cap is not None and n > cap:
            n = cap
        resolved.append(n)
    out = sorted({n for n in resolved if n > 0})
    if not out:
        raise ValueError("train_sizes resuelto vacío tras aplicar max/caps")
    return out


def get_learning_curve_train_sizes(
    *,
    cli_sizes: Optional[Sequence[Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    cell_id: Optional[str] = None,
    use_manifest_if_no_cli: bool = True,
) -> List[int]:
    """
    Tamaños resueltos para train/eval/summary.
    Sin CLI: usa train_sizes del manifiesto (ya resueltos tras preflight).
    Con CLI: parsea specs (incl. max) usando split maestro si existe.
    """
    manifest = load_learning_curve_manifest() if use_manifest_if_no_cli else None

    if cli_sizes is None and manifest and manifest.get("train_sizes"):
        return [int(x) for x in manifest["train_sizes"]]

    specs = parse_train_size_specs(cli_sizes, config)
    max_clips: Optional[int] = None
    if manifest and manifest.get("master_train_clips_global") is not None:
        max_clips = int(manifest["master_train_clips_global"])
    elif manifest and manifest.get("by_cell") and cell_id:
        cell_block = (manifest.get("by_cell") or {}).get(cell_id) or {}
        if cell_block.get("master_train_clips") is not None:
            max_clips = int(cell_block["master_train_clips"])
    if max_clips is None and cell_id:
        max_clips = master_train_clip_count(cell_id)
    if max_clips is None and manifest:
        by_cell = manifest.get("by_cell") or manifest.get("cells") or {}
        if isinstance(by_cell, dict) and by_cell:
            max_clips = max(
                int(v.get("master_train_clips", 0))
                for v in by_cell.values()
                if isinstance(v, dict)
            )
    return resolve_train_size_specs(specs, max_train_clips=max_clips)


def resolve_train_sizes(
    sizes: Optional[Sequence[Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    *,
    max_train_clips: Optional[int] = None,
    cell_id: Optional[str] = None,
) -> List[int]:
    """Compat: resuelve specs; prefer get_learning_curve_train_sizes en código nuevo."""
    if sizes is None and cell_id and max_train_clips is None:
        return get_learning_curve_train_sizes(config=config, cell_id=cell_id)
    specs = parse_train_size_specs(sizes, config)
    if max_train_clips is None and cell_id:
        max_train_clips = master_train_clip_count(cell_id)
    return resolve_train_size_specs(specs, max_train_clips=max_train_clips)


def resolve_learning_curve_cells(
    config: Dict[str, Any],
    cell_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Celdas de la curva: mc full/filtered + bin full/filtered (hardened) por defecto."""
    try:
        from campaign_paths import filter_cells
    except ImportError:
        from .campaign_paths import filter_cells  # type: ignore

    if cell_ids:
        cells = filter_cells(config, cell_ids)
        if not cells:
            raise ValueError(f"Celdas no encontradas en config: {cell_ids}")
        return cells

    lc = learning_curve_config(config)
    configured = lc.get("cells")
    if configured is not None:
        if configured == "all" or configured == ["all"]:
            return filter_cells(config, None)
        ids = [str(x) for x in configured]
        cells = filter_cells(config, ids)
        if not cells:
            raise ValueError(f"learning_curve.cells sin celdas válidas: {ids}")
        return cells

    legacy = lc.get("cell")
    if legacy:
        return filter_cells(config, [str(legacy)])

    cells = filter_cells(config, list(DEFAULT_LEARNING_CURVE_CELL_IDS))
    if not cells:
        raise ValueError(
            "No hay celdas learning_curve por defecto. Indica --cells o learning_curve.cells en config."
        )
    return cells


def resolve_learning_curve_cell(
    config: Dict[str, Any],
    cell_ids: Optional[List[str]] = None,
) -> str:
    """Compat: una sola celda (error si hay varias)."""
    cells = resolve_learning_curve_cells(config, cell_ids)
    if len(cells) != 1:
        raise ValueError(
            f"Se esperaba una celda, got {len(cells)}. Usa resolve_learning_curve_cells o --cells id1 id2 ..."
        )
    return str(cells[0]["id"])


def _task_experiment_id_spec(cell: Dict[str, Any], config: Dict[str, Any]) -> Any:
    """Resuelve spec de experiment_ids según task (binary/multiclass).

    Prioridad: mass_augment → raíz del config → learning_curve → experiment_ids global.
    """
    task = str(cell.get("task") or "")
    key = "binary_experiment_ids" if task == "binary" else "multiclass_experiment_ids"
    sources = (
        config.get("mass_augment") or {},
        config,
        learning_curve_config(config),
    )
    for source in sources:
        spec = source.get(key)
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
    from campaign_paths import resolve_experiment_ids

    return resolve_experiment_ids(_task_experiment_id_spec(cell, config))


def load_learning_curve_cell_ids(
    config: Optional[Dict[str, Any]] = None,
    cli_cell_ids: Optional[List[str]] = None,
) -> List[str]:
    if cli_cell_ids:
        return list(cli_cell_ids)
    manifest = load_learning_curve_manifest()
    if manifest and manifest.get("cell_ids"):
        return [str(x) for x in manifest["cell_ids"]]
    if config:
        return [c["id"] for c in resolve_learning_curve_cells(config, None)]
    return list(DEFAULT_LEARNING_CURVE_CELL_IDS)


def run_id_for_train_size(n: int) -> str:
    return f"lc_{int(n)}"


def learning_curve_manifest_path(master_run_id: str = LEARNING_CURVE_MASTER_RUN_ID) -> Path:
    return artifacts_root(master_run_id) / MANIFEST_FILENAME


def _build_uid_category_map(
    examples: List[Any],
    uids: Sequence[str],
) -> Dict[str, int]:
    uid_set = set(uids)
    out: Dict[str, int] = {}
    for ex in examples:
        uid = _example_uid(ex)
        if uid in uid_set:
            out[uid] = int(_example_folder_category(ex))
    return out


def subsample_train_uids(
    train_uids: List[str],
    uid_to_category: Dict[str, int],
    target_n: int,
    *,
    robbery_class: int,
    seed: int = SEED,
    keep_all_robbery: bool = True,
) -> List[str]:
    """Submuestreo estratificado del train; val/test no cambian."""
    target_n = int(target_n)
    if target_n <= 0:
        raise ValueError(f"target_n debe ser > 0 (got {target_n})")
    if target_n >= len(train_uids):
        return sorted(train_uids)

    rng = random.Random(seed + target_n)
    robbery_uids = sorted(u for u in train_uids if uid_to_category.get(u) == robbery_class)
    neg_uids = sorted(u for u in train_uids if uid_to_category.get(u) != robbery_class)

    if keep_all_robbery and len(robbery_uids) >= target_n:
        return sorted(rng.sample(robbery_uids, target_n))

    selected: List[str] = list(robbery_uids) if keep_all_robbery else []
    remaining = target_n - len(selected)
    if remaining <= 0:
        return sorted(selected[:target_n])

    by_cat: Dict[int, List[str]] = {}
    for u in neg_uids:
        by_cat.setdefault(uid_to_category.get(u, -1), []).append(u)

    total_neg = sum(len(v) for v in by_cat.values())
    if total_neg == 0:
        return sorted(selected[:target_n])

    neg_selected: List[str] = []
    cats = sorted(by_cat.keys())
    quotas: Dict[int, int] = {}
    assigned = 0
    for i, cat in enumerate(cats):
        if i == len(cats) - 1:
            q = remaining - assigned
        else:
            q = int(round(remaining * len(by_cat[cat]) / total_neg))
        quotas[cat] = max(0, q)
        assigned += quotas[cat]

    diff = remaining - assigned
    if diff != 0 and cats:
        quotas[cats[0]] = max(0, quotas[cats[0]] + diff)

    for cat in cats:
        pool = by_cat[cat][:]
        rng.shuffle(pool)
        take = min(quotas.get(cat, 0), len(pool))
        neg_selected.extend(pool[:take])

    if len(neg_selected) < remaining:
        already = set(neg_selected)
        rest = [u for u in neg_uids if u not in already]
        rng.shuffle(rest)
        neg_selected.extend(rest[: remaining - len(neg_selected)])

    selected.extend(neg_selected[:remaining])
    return sorted(selected[:target_n])


def _load_examples_for_plan(
    master_plan: Dict[str, Any],
    *,
    task: str,
    positive_class: int,
    class_map_spec: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    data_root = Path(master_plan["data_root"])
    filters = master_plan.get("filters") or {}
    examples = collect_examples(
        pose_source=str(master_plan["pose_source"]),
        single_user_only=bool(master_plan.get("single_user_only", True)),
        min_clip_seconds=float(filters.get("min_clip_seconds", 0.0)),
        min_valid_frames=int(filters.get("min_valid_frames", 0)),
        min_valid_pct=float(filters.get("min_valid_pct", 0.0)),
        max_occlusion_ratio=float(filters.get("max_occlusion_ratio", 1.0)),
        data_root=data_root,
    )
    if class_map_spec:
        try:
            from class_map_utils import apply_class_map_spec
        except ImportError:
            from .class_map_utils import apply_class_map_spec  # type: ignore
        examples = apply_class_map_spec(examples, class_map_spec)
    if task == "binary":
        examples = make_binary_examples(examples, positive_class=positive_class)
    return examples


def derive_plan_for_train_size(
    master_plan: Dict[str, Any],
    *,
    target_train_clips: int,
    examples: List[Any],
    robbery_class: int,
    negative_to_robbery_ratio: float,
    category_aug_config_path: Path,
    keep_all_robbery: bool = True,
    run_id: Optional[str] = None,
    cell_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Plan derivado: mismo val/test que master, train submuestreado."""
    master_uids = master_plan.get("split_uids") or {}
    train_uids_full = list(master_uids.get("train") or [])
    val_uids = list(master_uids.get("val") or [])
    test_uids = list(master_uids.get("test") or [])

    uid_to_cat = _build_uid_category_map(examples, train_uids_full)
    train_uids_sub = subsample_train_uids(
        train_uids_full,
        uid_to_cat,
        int(target_train_clips),
        robbery_class=int(robbery_class),
        seed=int(master_plan.get("seed", SEED)),
        keep_all_robbery=keep_all_robbery,
    )

    split_uids = {
        "train": train_uids_sub,
        "val": val_uids,
        "test": test_uids,
    }

    train_ex, val_ex, test_ex = split_examples_by_uid_manifest(examples, split_uids)
    train_counts = count_examples_by_folder_category(train_ex)
    val_counts = count_examples_by_folder_category(val_ex)
    test_counts = count_examples_by_folder_category(test_ex)

    cfg_path = Path(category_aug_config_path)
    current_cfg = load_category_augmentation_config(cfg_path)
    include_identity = bool(current_cfg.get("include_identity", True))
    max_aug_ops = int((master_plan.get("augmentation") or {}).get("max_ops_per_clip") or max_category_augment_ops_available())
    max_aug_ops = min(max_aug_ops, max_category_augment_ops_available())

    proposed_aug = propose_category_augment_counts(
        train_counts,
        robbery_class=int(robbery_class),
        negative_to_robbery_ratio=float(negative_to_robbery_ratio),
        max_aug=max_aug_ops,
        include_identity=include_identity,
    )

    proposed_cfg = copy.deepcopy(master_plan.get("proposed_category_augmentation") or {})
    proposed_cfg["categories"] = {str(k): int(v) for k, v in sorted(proposed_aug.items())}
    proposed_cfg.setdefault("include_identity", include_identity)
    proposed_cfg.setdefault("default", int(current_cfg.get("default", 0) or 0))

    proposed_rows = summarize_category_aug_on_train(train_counts, proposed_cfg)
    balance_rob = analyze_robbery_augment_balance(
        train_counts,
        proposed_aug,
        robbery_class=int(robbery_class),
        include_identity=include_identity,
        negative_to_robbery_ratio=float(negative_to_robbery_ratio),
    )
    dataset_stats = build_plan_stats_by_category(train_ex, val_ex, test_ex, proposed_cfg)
    totals = dataset_stats["totals"]

    plan = copy.deepcopy(master_plan)
    plan["created_at"] = datetime.now(timezone.utc).isoformat()
    plan["split_uids"] = split_uids
    plan["split_stats_by_category"] = {
        "train": {str(k): int(v) for k, v in train_counts.items()},
        "val": {str(k): int(v) for k, v in val_counts.items()},
        "test": {str(k): int(v) for k, v in test_counts.items()},
    }
    plan["proposed_category_augmentation"] = proposed_cfg
    plan["dataset_stats"] = dataset_stats
    plan["balance"]["robbery_balance"] = balance_rob
    plan["totals"] = {
        "examples_valid": int(master_plan.get("totals", {}).get("examples_valid", len(examples))),
        "uids_train": len(train_uids_sub),
        "uids_val": len(val_uids),
        "uids_test": len(test_uids),
        "rows_train_proposed": int(totals["rows_total_train"]),
        "rows_synthetic_train": int(totals["rows_synthetic_train"]),
    }
    plan["learning_curve"] = {
        "master_run_id": LEARNING_CURVE_MASTER_RUN_ID,
        "target_train_clips": int(target_train_clips),
        "train_clips_full": len(train_uids_full),
        "keep_all_robbery": bool(keep_all_robbery),
        "run_id": run_id,
        "cell_id": cell_id,
    }
    if plan.get("campaign") and run_id:
        plan["campaign"] = dict(plan["campaign"])
        plan["campaign"]["run_id"] = run_id
        plan["campaign"]["learning_curve"] = {
            "target_train_clips": int(target_train_clips),
            "master_run_id": LEARNING_CURVE_MASTER_RUN_ID,
        }

    meta = {
        "target_train_clips": int(target_train_clips),
        "train_clips_selected": len(train_uids_sub),
        "train_clips_full": len(train_uids_full),
        "val_clips": len(val_uids),
        "test_clips": len(test_uids),
        "rows_train_proposed": int(totals["rows_total_train"]),
        "robbery_train_clips": int(train_counts.get(int(robbery_class), 0)),
        "proposed_rows_by_category": {str(k): v for k, v in proposed_rows.items()},
    }
    return plan, meta


def write_learning_curve_manifest(
    payload: Dict[str, Any],
    master_run_id: str = LEARNING_CURVE_MASTER_RUN_ID,
) -> Path:
    path = learning_curve_manifest_path(master_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def resolve_ensemble_settings(
    config: Dict[str, Any],
    cell_id: Optional[str] = None,
    *,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    from campaign_paths import artifacts_root, resolve_experiment_ids

    lc = learning_curve_config(config)
    ens = dict(lc.get("ensemble") or {})
    cid = str(cell_id or lc.get("cell") or "bin_full_hardened")
    split = str(lc.get("eval_split") or ens.get("split") or "val")

    if ens.get("auto_from_eval") and run_id:
        from evaluate_campaign import load_best_ensemble_spec

        reports_dir = artifacts_root(run_id) / "reports" / cid
        spec = load_best_ensemble_spec(reports_dir, split)
        if spec:
            thrs = spec.get("thresholds")
            thr = spec.get("threshold")
            if isinstance(thr, list) and thr:
                threshold = float(thr[0])
            elif thr is not None:
                threshold = float(thr)
            elif isinstance(thrs, list) and thrs:
                threshold = float(thrs[0])
            elif isinstance(thrs, str) and thrs.strip():
                threshold = float(str(thrs).split("|")[0])
            else:
                threshold = 0.5
            return {
                "cell": cid,
                "models": list(spec.get("models") or []),
                "rule": str(spec.get("rule") or "mean"),
                "threshold": threshold,
                "split": split,
                "experiment_ids": resolve_experiment_ids(
                    lc.get("binary_experiment_ids") or lc.get("experiment_ids") or "all"
                ),
                "source": "best_ensemble.json",
            }

    models = list(ens.get("models") or [])
    return {
        "cell": cid,
        "models": models,
        "rule": str(ens.get("rule") or "mean"),
        "threshold": float(ens.get("threshold", 0.68)),
        "split": split,
        "experiment_ids": resolve_experiment_ids(
            lc.get("binary_experiment_ids") or lc.get("experiment_ids") or "all"
        ),
        "source": "campaign_config",
    }


def _derive_plans_for_cell(
    cell: Dict[str, Any],
    config: Dict[str, Any],
    *,
    master_plan: Dict[str, Any],
    master_plan_path: Path,
    resolved_sizes: List[int],
    examples: List[Any],
    write: bool = False,
    keep_all_robbery: bool = True,
) -> Dict[str, Any]:
    cell_id = cell["id"]
    master_run_id = LEARNING_CURVE_MASTER_RUN_ID
    master_train_clips = len((master_plan.get("split_uids") or {}).get("train") or [])

    derived = _derive_plans_for_cell_inner(
        cell,
        config,
        master_plan=master_plan,
        master_plan_path=master_plan_path,
        resolved_sizes=resolved_sizes,
        examples=examples,
        write=write,
        keep_all_robbery=keep_all_robbery,
        master_run_id=master_run_id,
    )
    derived["master_train_clips"] = master_train_clips
    return derived


def _derive_plans_for_cell_inner(
    cell: Dict[str, Any],
    config: Dict[str, Any],
    *,
    master_plan: Dict[str, Any],
    master_plan_path: Path,
    resolved_sizes: List[int],
    examples: List[Any],
    write: bool,
    keep_all_robbery: bool,
    master_run_id: str,
) -> Dict[str, Any]:
    cell_id = cell["id"]
    robbery_class = int(config.get("robbery_class", ROBBERY_CLASS))
    from preflight_campaign import _aug_profile as _campaign_aug_profile

    aug_id = str(master_plan.get("campaign", {}).get("aug_profile", "baseline"))
    aug_prof = _campaign_aug_profile(config, aug_id)
    neg_ratio = float(aug_prof.get("negative_to_robbery_ratio", 4.0))
    aug_path_master = category_aug_path(cell_id, run_id=master_run_id)
    master_train_clips = len((master_plan.get("split_uids") or {}).get("train") or [])

    size_rows: List[Dict[str, Any]] = []
    for n in resolved_sizes:
        run_id = run_id_for_train_size(n)
        print(f"\n--- [{cell_id}] train={n} → run_id={run_id} ---")
        derived_plan, meta = derive_plan_for_train_size(
            master_plan,
            target_train_clips=n,
            examples=examples,
            robbery_class=robbery_class,
            negative_to_robbery_ratio=neg_ratio,
            category_aug_config_path=aug_path_master,
            keep_all_robbery=keep_all_robbery,
            run_id=run_id,
            cell_id=cell_id,
        )
        row: Dict[str, Any] = {"train_size": n, "run_id": run_id, **meta}
        if write:
            arts = ensure_cell_dirs(cell_id, run_id=run_id)
            aug_out = category_aug_path(cell_id, run_id=run_id)
            proposed = derived_plan.get("proposed_category_augmentation") or {}
            with open(aug_out, "w", encoding="utf-8") as f:
                json.dump(proposed, f, indent=2, ensure_ascii=False)
                f.write("\n")
            derived_plan["category_augmentation_config"] = str(aug_out.resolve())
            plan_out = training_plan_path(cell_id, run_id=run_id)
            write_training_plan(derived_plan, plan_out)
            row["plan_path"] = str(plan_out)
            row["aug_path"] = str(aug_out)
            row["models_dir"] = str(arts["models_dir"])
            print(
                f"  [OK] clips={meta['train_clips_selected']}/{master_train_clips} "
                f"rows={meta['rows_train_proposed']}"
            )
        size_rows.append(row)

    return {
        "cell_id": cell_id,
        "task": cell.get("task"),
        "pose_source": cell.get("pose_source"),
        "master_run_id": master_run_id,
        "master_plan_path": str(master_plan_path),
        "sizes": size_rows,
    }


def run_learning_curve_preflight(
    cell: Dict[str, Any],
    config: Dict[str, Any],
    *,
    train_size_specs: Optional[Sequence[Any]] = None,
    train_sizes: Optional[List[int]] = None,
    resolved_sizes: Optional[List[int]] = None,
    data_root: Optional[Path] = None,
    write: bool = False,
    skip_time_estimate: bool = False,
    experiment_ids: Optional[List[int]] = None,
    keep_all_robbery: Optional[bool] = None,
) -> Dict[str, Any]:
    """Una celda: plan maestro + planes lc_<N>."""
    from preflight_campaign import run_preflight_cell

    cell_id = cell["id"]
    lc_cfg = learning_curve_config(config)
    keep_rob = bool(
        keep_all_robbery
        if keep_all_robbery is not None
        else lc_cfg.get("keep_all_robbery", True)
    )
    exp_ids = experiment_ids or experiment_ids_for_cell(cell, config)
    master_run_id = LEARNING_CURVE_MASTER_RUN_ID

    print(f"\n=== Learning curve — celda {cell_id} ===")
    master_row = run_preflight_cell(
        cell,
        config,
        data_root=data_root,
        write=write,
        skip_time_estimate=skip_time_estimate,
        experiment_ids=exp_ids,
        run_id=master_run_id,
    )
    if "error" in master_row:
        return {"cell_id": cell_id, "error": master_row["error"], "sizes": []}

    master_plan_path = training_plan_path(cell_id, run_id=master_run_id)
    with open(master_plan_path, "r", encoding="utf-8") as f:
        master_plan = json.load(f)

    master_train_clips = len((master_plan.get("split_uids") or {}).get("train") or [])
    specs = parse_train_size_specs(train_size_specs, config) if train_size_specs is not None else (
        parse_train_size_specs(None, config) if train_sizes is None else list(train_sizes)
    )
    if resolved_sizes is not None:
        sizes = list(resolved_sizes)
    elif train_sizes is not None and train_size_specs is None:
        sizes = sorted(set(int(n) for n in train_sizes))
    else:
        sizes = resolve_train_size_specs(specs, max_train_clips=master_train_clips)

    print(f"  Train maestro: {master_train_clips} | specs {specs} → {sizes}")

    try:
        from class_map_utils import load_class_map
        from campaign_paths import class_map_path
    except ImportError:
        from .class_map_utils import load_class_map  # type: ignore
        from .campaign_paths import class_map_path  # type: ignore

    class_map_spec = load_class_map(class_map_path(cell["class_map_id"]))
    examples = _load_examples_for_plan(
        master_plan,
        task=str(cell["task"]),
        positive_class=int(config.get("robbery_class", ROBBERY_CLASS)),
        class_map_spec=class_map_spec,
    )

    derived = _derive_plans_for_cell(
        cell,
        config,
        master_plan=master_plan,
        master_plan_path=master_plan_path,
        resolved_sizes=sizes,
        examples=examples,
        write=write,
        keep_all_robbery=keep_rob,
    )
    return {
        **derived,
        "train_size_specs": [str(s) for s in specs],
        "train_sizes": sizes,
        "master_train_clips": master_train_clips,
        "master": master_row,
    }


def run_learning_curve_preflight_all(
    cells: List[Dict[str, Any]],
    config: Dict[str, Any],
    *,
    train_size_specs: Optional[Sequence[Any]] = None,
    data_root: Optional[Path] = None,
    write: bool = False,
    skip_time_estimate: bool = False,
) -> Dict[str, Any]:
    """Todas las celdas: split maestro por celda + mismos tamaños (max = mayor train disponible)."""
    from preflight_campaign import run_preflight_cell

    lc_cfg = learning_curve_config(config)
    keep_rob = bool(lc_cfg.get("keep_all_robbery", True))
    specs = parse_train_size_specs(train_size_specs, config)
    master_run_id = LEARNING_CURVE_MASTER_RUN_ID

    print(f"\n=== Learning curve — {len(cells)} celdas ===")
    print(f"  Celdas: {[c['id'] for c in cells]}")
    print(f"  Specs train: {specs}")

    masters: Dict[str, Dict[str, Any]] = {}
    global_max = 0

    for cell in cells:
        cell_id = cell["id"]
        print(f"\n--- Master split: {cell_id} ---")
        exp_ids = experiment_ids_for_cell(cell, config)
        master_row = run_preflight_cell(
            cell,
            config,
            data_root=data_root,
            write=write,
            skip_time_estimate=skip_time_estimate,
            experiment_ids=exp_ids,
            run_id=master_run_id,
        )
        if "error" in master_row:
            masters[cell_id] = {"error": master_row["error"]}
            continue

        master_plan_path = training_plan_path(cell_id, run_id=master_run_id)
        with open(master_plan_path, "r", encoding="utf-8") as f:
            master_plan = json.load(f)
        n_train = len((master_plan.get("split_uids") or {}).get("train") or [])
        global_max = max(global_max, n_train)
        masters[cell_id] = {
            "master_row": master_row,
            "master_plan": master_plan,
            "master_plan_path": master_plan_path,
            "master_train_clips": n_train,
        }
        print(f"  [{cell_id}] train maestro: {n_train} clips")

    if global_max <= 0:
        raise RuntimeError("No se pudo generar ningún split maestro para learning curve")

    resolved_sizes = resolve_train_size_specs(specs, max_train_clips=global_max)
    print(f"\n  Train global max: {global_max} → tamaños resueltos: {resolved_sizes}")

    by_cell: Dict[str, Any] = {}
    errors: List[str] = []
    for cell in cells:
        cell_id = cell["id"]
        block = masters.get(cell_id) or {}
        if block.get("error"):
            errors.append(f"{cell_id}: {block['error']}")
            by_cell[cell_id] = block
            continue

        try:
            from class_map_utils import load_class_map
            from campaign_paths import class_map_path
        except ImportError:
            from .class_map_utils import load_class_map  # type: ignore
            from .campaign_paths import class_map_path  # type: ignore

        class_map_spec = load_class_map(class_map_path(cell["class_map_id"]))
        examples = _load_examples_for_plan(
            block["master_plan"],
            task=str(cell["task"]),
            positive_class=int(config.get("robbery_class", ROBBERY_CLASS)),
            class_map_spec=class_map_spec,
        )
        derived = _derive_plans_for_cell(
            cell,
            config,
            master_plan=block["master_plan"],
            master_plan_path=Path(block["master_plan_path"]),
            resolved_sizes=resolved_sizes,
            examples=examples,
            write=write,
            keep_all_robbery=keep_rob,
        )
        by_cell[cell_id] = {**block, **derived}

    manifest = {
        "mode": "learning_curve",
        "cell_ids": [c["id"] for c in cells],
        "master_run_id": master_run_id,
        "train_size_specs": [str(s) for s in specs],
        "train_sizes": resolved_sizes,
        "master_train_clips_global": global_max,
        "keep_all_robbery": keep_rob,
        "by_cell": by_cell,
        "ensemble": resolve_ensemble_settings(config),
        "errors": errors,
    }
    if write:
        mpath = write_learning_curve_manifest(manifest, master_run_id)
        print(f"\nManifiesto learning curve → {mpath}")

    return {
        "cell_ids": [c["id"] for c in cells],
        "train_size_specs": [str(s) for s in specs],
        "train_sizes": resolved_sizes,
        "master_train_clips_global": global_max,
        "by_cell": by_cell,
        "errors": errors,
        "manifest_path": str(learning_curve_manifest_path(master_run_id)),
    }

