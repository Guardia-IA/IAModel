#!/usr/bin/env python3
"""
Preflight completo antes de entrenar: inventario, balance, augment máximo por categoría,
split estratificado por UID y artefacto training_plan.json reutilizable en train.

Uso:
  python preflight_train_plan.py --task multiclass --single-user-only
  python preflight_train_plan.py --task binary --write-plan --write-config
  python preflight_train_plan.py --task binary --max-aug 15 --write-plan
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .model_config import (  # type: ignore[attr-defined]
        MIN_CLIP_SECONDS,
        MIN_VALID_FRAMES,
        MIN_VALID_PCT,
        MAX_OCCLUSION_RATIO,
        SEED,
        CATEGORY_AUGMENTATION_CONFIG_PATH,
        TRAINING_PLAN_PATH,
        ROBBERY_CLASS,
        PREFLIGHT_MIN_ROBBERY_TRAIN_ROWS,
        PREFLIGHT_MIN_NEGATIVE_TRAIN_ROWS,
        PREFLIGHT_NEGATIVE_TO_ROBBERY_RATIO,
        DEFAULT_BINARY_SOFTMAX_THRESHOLD,
        DEFAULT_BINARY_LOGIT_MARGIN,
        suggest_split_ratios,
    )
    from .train_model_operations import (  # type: ignore[attr-defined]
        CATEGORY_AUGMENT_RECIPE_BANK,
        collect_examples,
        split_examples_stratified_by_uid,
        split_uids_from_example_lists,
        get_data_result_root,
        scan_data_result_folders,
        load_category_augmentation_config,
        count_examples_by_folder_category,
        propose_category_augment_counts,
        summarize_category_aug_on_train,
        analyze_robbery_augment_balance,
        build_plan_stats_by_category,
        max_category_augment_ops_available,
        _category_augment_count_for_label,
        _category_augment_is_active,
        make_binary_examples,
    )
except ImportError:
    from model_config import (  # type: ignore[attr-defined]
        MIN_CLIP_SECONDS,
        MIN_VALID_FRAMES,
        MIN_VALID_PCT,
        MAX_OCCLUSION_RATIO,
        SEED,
        CATEGORY_AUGMENTATION_CONFIG_PATH,
        TRAINING_PLAN_PATH,
        ROBBERY_CLASS,
        PREFLIGHT_MIN_ROBBERY_TRAIN_ROWS,
        PREFLIGHT_MIN_NEGATIVE_TRAIN_ROWS,
        PREFLIGHT_NEGATIVE_TO_ROBBERY_RATIO,
        DEFAULT_BINARY_SOFTMAX_THRESHOLD,
        DEFAULT_BINARY_LOGIT_MARGIN,
        suggest_split_ratios,
    )
    from train_model_operations import (  # type: ignore[attr-defined]
        CATEGORY_AUGMENT_RECIPE_BANK,
        collect_examples,
        split_examples_stratified_by_uid,
        split_uids_from_example_lists,
        get_data_result_root,
        scan_data_result_folders,
        load_category_augmentation_config,
        count_examples_by_folder_category,
        propose_category_augment_counts,
        summarize_category_aug_on_train,
        analyze_robbery_augment_balance,
        build_plan_stats_by_category,
        max_category_augment_ops_available,
        _category_augment_count_for_label,
        _category_augment_is_active,
        make_binary_examples,
    )

try:
    from .training_time_estimate import (  # type: ignore[attr-defined]
        estimate_all_experiments,
        format_estimate_report,
        fmt_duration,
    )
except ImportError:
    from training_time_estimate import (  # type: ignore[attr-defined]
        estimate_all_experiments,
        format_estimate_report,
        fmt_duration,
    )

try:
    from .training_artifacts import resolve_artifacts, print_artifact_banner  # type: ignore[attr-defined]
except ImportError:
    from training_artifacts import resolve_artifacts, print_artifact_banner  # type: ignore[attr-defined]

RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"


def header(title: str) -> None:
    line = "=" * 72
    print(f"\n{BOLD}{line}\n{title}\n{line}{RESET}")


def _pct(n: int, total: int) -> float:
    return 100.0 * n / total if total > 0 else 0.0


def _imbalance_ratio(counts: Dict[int, int]) -> float:
    vals = [v for v in counts.values() if v > 0]
    if not vals:
        return 0.0
    return max(vals) / max(1, min(vals))


def _build_proposed_config(
    proposed: Dict[int, int],
    *,
    include_identity: bool = True,
    default: int = 0,
    robbery_class: int = ROBBERY_CLASS,
) -> Dict[str, Any]:
    return {
        "enabled": True,
        "include_identity": include_identity,
        "default": int(default),
        "categories": {str(k): int(v) for k, v in sorted(proposed.items(), key=lambda x: int(x[0]))},
        "_comment": (
            f"Generado por preflight_train_plan.py. Clase {robbery_class}=robo. "
            "Solo train; val/test sin augment."
        ),
    }


def _merge_train_counts_with_folders(
    train_counts: Dict[int, int],
    folder_scan: Dict[int, Dict[str, int]],
) -> Dict[int, int]:
    merged = {int(k): int(v) for k, v in train_counts.items()}
    for cat in folder_scan:
        merged.setdefault(int(cat), 0)
    return dict(sorted(merged.items()))


def _print_balance_summary(
    global_counts: Dict[int, int],
    train_counts: Dict[int, int],
    proposed_rows: Dict[int, Dict[str, int]],
    imbalance_raw: float,
    imbalance_after: float,
) -> None:
    header("4) Balance por categoría (real vs sintético en train)")
    total_real = sum(train_counts.values())
    total_rows = sum(v["train_rows"] for v in proposed_rows.values())
    total_synthetic = total_rows - total_real
    print(f"  Clips reales en train: {CYAN}{total_real}{RESET}")
    print(f"  Filas train tras augment propuesto: {GREEN}{total_rows}{RESET}")
    print(f"    → reales: {total_real} | sintéticas (augment): {total_synthetic}")
    print(f"  Ratio desbalance clips (max/min): {imbalance_raw:.2f}x")
    print(f"  Ratio desbalance filas train (max/min cat): {imbalance_after:.2f}x")
    if imbalance_raw > 5.0:
        print(f"  {YELLOW}[!] Desbalance alto en bruto; el augment propuesto intenta equilibrar filas.{RESET}")
    elif imbalance_raw <= 2.0:
        print(f"  {GREEN}Balance en bruto razonable (≤2x).{RESET}")

    print(f"\n{'cat':>4} | {'real':>5} | {'aug/c':>5} | {'rows':>6} | {'sint':>5}")
    print("-" * 38)
    cats = sorted(set(global_counts) | set(train_counts), key=lambda x: int(x))
    for cat in cats:
        real = int(train_counts.get(cat, 0))
        detail = proposed_rows.get(cat, {})
        aug = int(detail.get("aug_per_clip", 0))
        rows = int(detail.get("train_rows", real))
        sint = max(0, rows - real)
        print(f"{cat:4d} | {real:5d} | {aug:5d} | {rows:6d} | {sint:5d}")


def _print_split_table(
    train_counts: Dict[int, int],
    val_counts: Dict[int, int],
    test_counts: Dict[int, int],
    split_uids: Dict[str, List[str]],
) -> None:
    header("3) Split train / val / test (estrato por categoría, UID único por clip)")
    n_tr = len(split_uids["train"])
    n_va = len(split_uids["val"])
    n_te = len(split_uids["test"])
    n_tot = n_tr + n_va + n_te
    print(
        f"  UIDs únicos => train: {GREEN}{n_tr}{RESET} ({_pct(n_tr, n_tot):.1f}%) | "
        f"val: {n_va} ({_pct(n_va, n_tot):.1f}%) | "
        f"test: {n_te} ({_pct(n_te, n_tot):.1f}%)"
    )
    print(f"  {GREEN}Garantía:{RESET} ningún UID aparece en más de un split (verificado).")
    print(f"  {YELLOW}Val/test: sin data augmentation.{RESET}")
    print(f"\n{'cat':>4} | {'train':>6} | {'val':>5} | {'test':>5}")
    print("-" * 32)
    cats = sorted(set(train_counts) | set(val_counts) | set(test_counts), key=lambda x: int(x))
    for cat in cats:
        print(
            f"{cat:4d} | {train_counts.get(cat, 0):6d} | "
            f"{val_counts.get(cat, 0):5d} | {test_counts.get(cat, 0):5d}"
        )


def build_training_plan(
    *,
    task: str = "multiclass",
    positive_class: int = ROBBERY_CLASS,
    pose_source: str = "filtered",
    single_user_only: bool = False,
    min_clip_seconds: float = MIN_CLIP_SECONDS,
    min_valid_frames: int = MIN_VALID_FRAMES,
    min_valid_pct: float = MIN_VALID_PCT,
    max_occlusion_ratio: float = MAX_OCCLUSION_RATIO,
    train_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
    data_root: Optional[Path] = None,
    category_aug_config: Path = CATEGORY_AUGMENTATION_CONFIG_PATH,
    target_samples: Optional[int] = None,
    max_aug: Optional[int] = None,
    robbery_class: int = ROBBERY_CLASS,
    min_robbery_rows: Optional[int] = None,
    min_negative_rows: Optional[int] = None,
    negative_to_robbery_ratio: float = PREFLIGHT_NEGATIVE_TO_ROBBERY_RATIO,
    binary_softmax_threshold: float = DEFAULT_BINARY_SOFTMAX_THRESHOLD,
    binary_logit_margin: float = DEFAULT_BINARY_LOGIT_MARGIN,
    skip_time_estimate: bool = False,
    class_map_spec: Optional[Dict[str, Any]] = None,
    experiment_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    resolved_root = get_data_result_root(data_root)
    folder_scan = scan_data_result_folders(resolved_root)

    header("1) Inventario data_result")
    print(f"  Raíz: {CYAN}{resolved_root}{RESET}")
    if not folder_scan:
        print(f"  {YELLOW}No hay categorías numéricas bajo data_result.{RESET}")

    examples = collect_examples(
        pose_source=pose_source,
        single_user_only=single_user_only,
        min_clip_seconds=float(min_clip_seconds),
        min_valid_frames=int(min_valid_frames),
        min_valid_pct=float(min_valid_pct),
        max_occlusion_ratio=float(max_occlusion_ratio),
        data_root=resolved_root,
    )
    if class_map_spec:
        try:
            from .class_map_utils import apply_class_map_spec, plan_class_map_block
        except ImportError:
            from class_map_utils import apply_class_map_spec, plan_class_map_block  # type: ignore
        n_before = len(examples)
        examples = apply_class_map_spec(examples, class_map_spec)
        print(
            f"  Class map {class_map_spec.get('id')}: "
            f"{n_before} → {len(examples)} ejemplos tras exclude/remap"
        )
    global_counts = count_examples_by_folder_category(examples)
    n_total = len(examples)
    print(f"  Ejemplos válidos: {CYAN}{n_total}{RESET} | categorías: {len(global_counts)}")

    if task == "binary":
        examples_for_split = make_binary_examples(examples, positive_class=positive_class)
        print(f"  Modo binario: clase {positive_class} (robo) → 1, resto → 0")
    else:
        examples_for_split = examples
        print("  Modo multiclass: categorías 0–14")

    if (train_ratio is None) ^ (val_ratio is None):
        raise ValueError("Indica ambos --train-ratio y --val-ratio, o ninguno.")
    if train_ratio is None or val_ratio is None:
        tr, vr, te = suggest_split_ratios(n_total)
    else:
        tr, vr = float(train_ratio), float(val_ratio)
        if tr + vr > 1.0:
            raise ValueError(f"train_ratio+val_ratio debe ser <= 1.0 (got {tr}+{vr})")
        te = 1.0 - tr - vr

    train_ex, val_ex, test_ex = split_examples_stratified_by_uid(
        examples_for_split, seed=SEED, train_ratio=tr, val_ratio=vr
    )
    split_uids = split_uids_from_example_lists(train_ex, val_ex, test_ex)

    train_counts = count_examples_by_folder_category(train_ex)
    val_counts = count_examples_by_folder_category(val_ex)
    test_counts = count_examples_by_folder_category(test_ex)
    train_counts = _merge_train_counts_with_folders(train_counts, folder_scan)

    _print_split_table(train_counts, val_counts, test_counts, split_uids)

    max_aug_ops = int(max_aug if max_aug is not None else max_category_augment_ops_available())
    max_aug_ops = min(max_aug_ops, max_category_augment_ops_available())

    cfg_path = Path(category_aug_config)
    current_cfg = load_category_augmentation_config(cfg_path)
    include_identity = bool(current_cfg.get("include_identity", True))

    proposed_aug = propose_category_augment_counts(
        train_counts,
        robbery_class=robbery_class,
        target_samples=target_samples,
        min_robbery_rows=min_robbery_rows,
        min_negative_rows=min_negative_rows,
        negative_to_robbery_ratio=negative_to_robbery_ratio,
        max_aug=max_aug_ops,
        include_identity=include_identity,
    )
    for cat in folder_scan:
        proposed_aug.setdefault(int(cat), 0)

    proposed_cfg = _build_proposed_config(
        proposed_aug,
        include_identity=include_identity,
        default=int(current_cfg.get("default", 0) or 0),
        robbery_class=robbery_class,
    )
    proposed_rows = summarize_category_aug_on_train(train_counts, proposed_cfg)
    balance_rob = analyze_robbery_augment_balance(
        train_counts,
        proposed_aug,
        robbery_class=robbery_class,
        include_identity=include_identity,
        negative_to_robbery_ratio=negative_to_robbery_ratio,
    )

    row_counts = [v["train_rows"] for v in proposed_rows.values() if v["train_rows"] > 0]
    imbalance_raw = _imbalance_ratio(train_counts)
    imbalance_after = _imbalance_ratio({k: v["train_rows"] for k, v in proposed_rows.items()})

    _print_balance_summary(
        global_counts, train_counts, proposed_rows, imbalance_raw, imbalance_after
    )

    dataset_stats = build_plan_stats_by_category(train_ex, val_ex, test_ex, proposed_cfg)

    header("5) Resumen del plan de entrenamiento")
    totals = dataset_stats["totals"]
    print(f"  Task: {task} | positive_class={positive_class}")
    print(f"  Split: {tr:.3f}/{vr:.3f}/{te:.3f} | seed={SEED}")
    print(f"  Augment máx ops/clip: {max_aug_ops} (banco={len(CATEGORY_AUGMENT_RECIPE_BANK)} recetas)")
    print(
        f"  Train filas totales propuestas: {GREEN}{totals['rows_total_train']}{RESET} "
        f"(reales {totals['clips_real_train']} + sintéticas {totals['rows_synthetic_train']})"
    )
    print(f"  Val clips reales: {totals['clips_real_val']} | Test clips reales: {totals['clips_real_test']}")
    if balance_rob.get("warnings"):
        for w in balance_rob["warnings"]:
            print(f"  {YELLOW}[!] {w}{RESET}")

    ops_available = sorted(
        {op["op"] for recipe in CATEGORY_AUGMENT_RECIPE_BANK for op in recipe.get("ops", [])}
    )

    plan: Dict[str, Any] = {
        "version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "task": task,
        "positive_class": int(positive_class),
        "campaign_class_map_id": (class_map_spec or {}).get("id"),
        "data_root": str(resolved_root),
        "pose_source": pose_source,
        "single_user_only": bool(single_user_only),
        "filters": {
            "min_clip_seconds": float(min_clip_seconds),
            "min_valid_frames": int(min_valid_frames),
            "min_valid_pct": float(min_valid_pct),
            "max_occlusion_ratio": float(max_occlusion_ratio),
        },
        "split_method": "stratified_by_folder_category_uid",
        "split_ratios": {"train": tr, "val": vr, "test": te},
        "split_uids": split_uids,
        "split_stats_by_category": {
            "train": {str(k): int(v) for k, v in train_counts.items()},
            "val": {str(k): int(v) for k, v in val_counts.items()},
            "test": {str(k): int(v) for k, v in test_counts.items()},
        },
        "balance": {
            "imbalance_ratio_clips_raw": float(imbalance_raw),
            "imbalance_ratio_rows_after_aug": float(imbalance_after),
            "robbery_balance": balance_rob,
        },
        "category_augmentation_config": str(cfg_path.resolve()),
        "proposed_category_augmentation": proposed_cfg,
        "augmentation": {
            "max_ops_per_clip": max_aug_ops,
            "recipe_bank_size": len(CATEGORY_AUGMENT_RECIPE_BANK),
            "include_identity": include_identity,
            "operations_available": ops_available,
            "recipe_names": [r.get("name", "?") for r in CATEGORY_AUGMENT_RECIPE_BANK],
        },
        "dataset_stats": dataset_stats,
        "evaluation": {
            "binary_softmax_threshold": float(binary_softmax_threshold),
            "binary_logit_margin": float(binary_logit_margin),
            "modes": ["softmax_argmax", "softmax_threshold", "logit_margin"],
        },
        "totals": {
            "examples_valid": n_total,
            "uids_train": len(split_uids["train"]),
            "uids_val": len(split_uids["val"]),
            "uids_test": len(split_uids["test"]),
            "rows_train_proposed": int(totals["rows_total_train"]),
            "rows_synthetic_train": int(totals["rows_synthetic_train"]),
        },
    }
    if class_map_spec:
        try:
            from .class_map_utils import plan_class_map_block
        except ImportError:
            from class_map_utils import plan_class_map_block  # type: ignore
        plan["class_map"] = plan_class_map_block(class_map_spec)

    if not skip_time_estimate:
        est_title = "7) Estimación de tiempo (todos los experimentos en model_config.py)"
        if experiment_ids:
            ids_str = ", ".join(str(i) for i in experiment_ids)
            est_title = f"7) Estimación de tiempo — experimentos [{ids_str}]"
        time_summary = estimate_all_experiments(
            train_rows=int(totals["rows_total_train"]),
            val_rows=int(totals["clips_real_val"]),
            experiment_ids=experiment_ids,
        )
        format_estimate_report(
            time_summary,
            bold=BOLD,
            reset=RESET,
            cyan=CYAN,
            yellow=YELLOW,
            header_fn=header,
            title=est_title,
        )
        plan["training_time_estimate"] = {
            "train_rows_per_epoch": time_summary["train_rows_per_epoch"],
            "val_rows_per_epoch": time_summary["val_rows_per_epoch"],
            "experiments_pending": time_summary["experiments_pending"],
            "experiments_skipped_done": time_summary["experiments_skipped_done"],
            "cuda_available": time_summary["cuda_available"],
            "total_gpu_seconds": time_summary["total_gpu_seconds"],
            "total_cpu_seconds": time_summary["total_cpu_seconds"],
            "primary_device": time_summary["primary_device"],
            "primary_total_seconds": time_summary["primary_total_seconds"],
            "primary_total_human": fmt_duration(time_summary["primary_total_seconds"]),
            "per_experiment": [
                {
                    "exp_id": item["exp_id"],
                    "arch": item["arch"],
                    "epochs": item["epochs"],
                    "batch_size": item["batch_size"],
                    "seq_len": item["seq_len"],
                    "gpu_minutes": round(item["gpu_seconds"] / 60.0, 2),
                    "cpu_minutes": round(item["cpu_seconds"] / 60.0, 2),
                }
                for item in time_summary["per_experiment"]
            ],
        }

    return plan


def write_training_plan(plan: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preflight completo: balance, augment, split UID y training_plan.json"
    )
    parser.add_argument("--task", choices=["multiclass", "binary"], default="multiclass")
    parser.add_argument("--positive-class", type=int, default=ROBBERY_CLASS)
    parser.add_argument("--pose-source", choices=["filtered", "full"], default="filtered")
    parser.add_argument("--single-user-only", action="store_true")
    parser.add_argument("--min-clip-seconds", type=float, default=MIN_CLIP_SECONDS)
    parser.add_argument("--min-valid-frames", type=int, default=MIN_VALID_FRAMES)
    parser.add_argument("--min-valid-pct", type=float, default=MIN_VALID_PCT)
    parser.add_argument("--max-occlusion-ratio", type=float, default=MAX_OCCLUSION_RATIO)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument(
        "--category-aug-config",
        type=str,
        default=None,
        help="JSON augment por categoría (default: multiclase o binario según --task).",
    )
    parser.add_argument("--target-samples", type=int, default=None)
    parser.add_argument(
        "--max-aug",
        type=int,
        default=None,
        help=f"Máx variantes augment por clip (default: {max_category_augment_ops_available()} recetas del banco).",
    )
    parser.add_argument("--min-robbery-rows", type=int, default=None)
    parser.add_argument("--min-negative-rows", type=int, default=None)
    parser.add_argument("--negative-to-robbery-ratio", type=float, default=PREFLIGHT_NEGATIVE_TO_ROBBERY_RATIO)
    parser.add_argument("--robbery-class", type=int, default=ROBBERY_CLASS)
    parser.add_argument(
        "--write-plan",
        action="store_true",
        help="Escribe el plan (default por task: training_plan.json o training_plan_binary.json).",
    )
    parser.add_argument(
        "--output-plan",
        type=str,
        default=None,
        help="Ruta del plan JSON (default según --task, no machaca el otro modo).",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Escribe config augment (default por task, archivo distinto en binario).",
    )
    parser.add_argument(
        "--output-config",
        type=str,
        default=None,
        help="Ruta config_category_augmentation_*.json (default según --task).",
    )
    parser.add_argument(
        "--binary-softmax-threshold",
        type=float,
        default=DEFAULT_BINARY_SOFTMAX_THRESHOLD,
    )
    parser.add_argument(
        "--binary-logit-margin",
        type=float,
        default=DEFAULT_BINARY_LOGIT_MARGIN,
    )
    parser.add_argument(
        "--skip-time-estimate",
        action="store_true",
        help="Omite la estimación de tiempo de todos los experimentos en model_config.py.",
    )
    args = parser.parse_args()

    artifacts = resolve_artifacts(
        args.task,
        single_user_only=args.single_user_only,
        training_plan=args.output_plan,
        category_aug_config=args.category_aug_config,
        mkdir=False,
    )
    print_artifact_banner(artifacts, title=f"Preflight ({args.task})")

    cat_aug_path = artifacts["category_aug_config"]
    data_root = Path(args.data_root) if args.data_root else None
    plan = build_training_plan(
        task=args.task,
        positive_class=args.positive_class,
        pose_source=args.pose_source,
        single_user_only=args.single_user_only,
        min_clip_seconds=args.min_clip_seconds,
        min_valid_frames=args.min_valid_frames,
        min_valid_pct=args.min_valid_pct,
        max_occlusion_ratio=args.max_occlusion_ratio,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        data_root=data_root,
        category_aug_config=cat_aug_path,
        target_samples=args.target_samples,
        max_aug=args.max_aug,
        robbery_class=args.robbery_class,
        min_robbery_rows=args.min_robbery_rows,
        min_negative_rows=args.min_negative_rows,
        negative_to_robbery_ratio=args.negative_to_robbery_ratio,
        binary_softmax_threshold=args.binary_softmax_threshold,
        binary_logit_margin=args.binary_logit_margin,
        skip_time_estimate=args.skip_time_estimate,
    )

    header("8) Siguiente paso")
    print(
        f"  {GREEN}1.{RESET} Revisa balance y augment arriba.\n"
        f"  {GREEN}2.{RESET} Genera artefactos:\n"
        f"       python preflight_train_plan.py --task {args.task}"
        + (" --single-user-only" if args.single_user_only else "")
        + " --write-plan --write-config\n"
        f"  {GREEN}3.{RESET} Entrena (artefactos separados si task=binary):\n"
        f"       python train_model_operations.py --task {args.task}"
        + (" --single-user-only" if args.single_user_only else "")
        + f" --training-plan {artifacts['training_plan']}"
    )

    if args.write_config:
        cfg_out = Path(args.output_config) if args.output_config else artifacts["category_aug_config"]
        proposed = plan["proposed_category_augmentation"]
        cfg_out.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_out, "w", encoding="utf-8") as f:
            json.dump(proposed, f, indent=2, ensure_ascii=False)
            f.write("\n")
        plan["category_augmentation_config"] = str(cfg_out.resolve())
        print(f"\n  {GREEN}[OK]{RESET} Config augment escrita: {cfg_out}")

    if args.write_plan:
        plan_path = Path(args.output_plan) if args.output_plan else artifacts["training_plan"]
        write_training_plan(plan, plan_path)
        print(f"  {GREEN}[OK]{RESET} training_plan escrito: {plan_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
