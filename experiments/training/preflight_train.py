#!/usr/bin/env python3
"""
Pre-chequeo antes de entrenar: inventario de clips por categoría, split train/val/test
y propuesta de augmentación por categoría (config_category_augmentation.json).

La data augmentation solo aplica a TRAIN. Val y test usan el clip original sin transformaciones.

Uso:
  python preflight_train.py
  python preflight_train.py --write-config
  python preflight_train.py --target-samples 60 --max-aug 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from .model_config import (  # type: ignore[attr-defined]
        MIN_CLIP_SECONDS,
        MIN_VALID_FRAMES,
        MIN_VALID_PCT,
        MAX_OCCLUSION_RATIO,
        SEED,
        CATEGORY_AUGMENTATION_CONFIG_PATH,
        ROBBERY_CLASS,
        PREFLIGHT_MIN_ROBBERY_TRAIN_ROWS,
        PREFLIGHT_MIN_NEGATIVE_TRAIN_ROWS,
        PREFLIGHT_NEGATIVE_TO_ROBBERY_RATIO,
        suggest_split_ratios,
    )
    from .train_model_operations import (  # type: ignore[attr-defined]
        collect_examples,
        split_examples,
        load_category_augmentation_config,
        count_examples_by_action_label,
        propose_category_augment_counts,
        summarize_category_aug_on_train,
        analyze_robbery_augment_balance,
        _category_augment_count_for_label,
        _category_augment_is_active,
    )
except ImportError:
    from model_config import (  # type: ignore[attr-defined]
        MIN_CLIP_SECONDS,
        MIN_VALID_FRAMES,
        MIN_VALID_PCT,
        MAX_OCCLUSION_RATIO,
        SEED,
        CATEGORY_AUGMENTATION_CONFIG_PATH,
        ROBBERY_CLASS,
        PREFLIGHT_MIN_ROBBERY_TRAIN_ROWS,
        PREFLIGHT_MIN_NEGATIVE_TRAIN_ROWS,
        PREFLIGHT_NEGATIVE_TO_ROBBERY_RATIO,
        suggest_split_ratios,
    )
    from train_model_operations import (  # type: ignore[attr-defined]
        collect_examples,
        split_examples,
        load_category_augmentation_config,
        count_examples_by_action_label,
        propose_category_augment_counts,
        summarize_category_aug_on_train,
        analyze_robbery_augment_balance,
        _category_augment_count_for_label,
        _category_augment_is_active,
    )

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


def _aug_map_for_train(train_counts: Dict[int, int], cfg: Dict[str, Any]) -> Dict[int, int]:
    return {int(cat): _category_augment_count_for_label(cfg, int(cat)) for cat in train_counts}


def _print_robbery_balance(
    balance: Dict[str, Any],
    robbery_class: int,
    min_robbery_rows: int,
    min_negative_rows: int,
    neg_ratio_target: float,
) -> None:
    header(f"4) Balance robo (clase {robbery_class}) vs resto (falsos positivos)")
    print(
        "  Prioridad: (1) no perder robos (FN) → augment clase 6 si hay pocos clips; "
        "(2) no disparar FP (normal→6) → augment negativos y ratio neg/robo alto."
    )
    print(
        f"  Clase {robbery_class} en train: {balance['robbery_clips_train']} clips → "
        f"{balance['robbery_rows_effective']} filas efectivas (obj. robo ≥ ~{min_robbery_rows})"
    )
    print(
        f"  Resto de clases: {balance['negative_rows_effective']} filas efectivas | "
        f"ratio neg/robo = {balance['negative_to_robbery_ratio']:.2f} "
        f"(objetivo ≥ {neg_ratio_target:.1f})"
    )
    print(
        f"  Share robo en train (bruto): {100.0 * balance['robbery_share_raw']:.1f}%"
    )
    for w in balance.get("warnings", []):
        print(f"  {YELLOW}[!]{RESET} {w}")
    if not balance.get("warnings"):
        print(f"  {GREEN}Balance augment razonable respecto a objetivos FP/FN.{RESET}")


def _print_cat_table(
    global_counts: Dict[int, int],
    train_counts: Dict[int, int],
    val_counts: Dict[int, int],
    test_counts: Dict[int, int],
    current_aug: Dict[int, int],
    proposed_aug: Dict[int, int],
    current_rows: Dict[int, Dict[str, int]],
    proposed_rows: Dict[int, Dict[str, int]],
) -> None:
    header("5) Detalle por categoría de acción")
    cats = sorted(
        set(global_counts) | set(train_counts) | set(val_counts) | set(test_counts),
        key=lambda x: int(x),
    )
    print(
        f"{'cat':>4} | {'total':>7} | {'train':>7} | {'val':>6} | {'test':>6} | "
        f"{'aug_now':>7} | {'aug_prop':>8} | {'rows_now':>8} | {'rows_prop':>9}"
    )
    print("-" * 88)
    total_train_rows_now = sum(v["train_rows"] for v in current_rows.values())
    total_train_rows_prop = sum(v["train_rows"] for v in proposed_rows.values())

    for cat in cats:
        g = global_counts.get(cat, 0)
        tr = train_counts.get(cat, 0)
        va = val_counts.get(cat, 0)
        te = test_counts.get(cat, 0)
        an = current_aug.get(cat, 0)
        ap = proposed_aug.get(cat, 0)
        rn = current_rows.get(cat, {}).get("train_rows", tr)
        rp = proposed_rows.get(cat, {}).get("train_rows", tr)
        print(
            f"{cat:4d} | {g:7d} | {tr:7d} | {va:6d} | {te:6d} | "
            f"{an:7d} | {ap:8d} | {rn:8d} | {rp:9d}"
        )

    print("-" * 88)
    print(
        f"{'Σ':>4} | {sum(global_counts.values()):7d} | {sum(train_counts.values()):7d} | "
        f"{sum(val_counts.values()):6d} | {sum(test_counts.values()):6d} | "
        f"{'':>7} | {'':>8} | {total_train_rows_now:8d} | {total_train_rows_prop:9d}"
    )
    print(
        f"\n  Desbalance train (max/min filas por cat): "
        f"actual={_imbalance_ratio({c: v['train_rows'] for c, v in current_rows.items()}):.2f}x | "
        f"propuesto={_imbalance_ratio({c: v['train_rows'] for c, v in proposed_rows.items()}):.2f}x"
    )
    print(
        f"\n  {YELLOW}rows_* = filas en train tras augment por categoría "
        f"(identidad + N variantes). Val/test = columnas val/test sin augment.{RESET}"
    )


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
            f"Generado por preflight_train.py. Clase {robbery_class}=robo (recall). "
            "Resto=reducir FP normal→robo. Solo train."
        ),
    }


def run_preflight(
    *,
    pose_source: str = "filtered",
    single_user_only: bool = False,
    min_clip_seconds: float = MIN_CLIP_SECONDS,
    min_valid_frames: int = MIN_VALID_FRAMES,
    min_valid_pct: float = MIN_VALID_PCT,
    max_occlusion_ratio: float = MAX_OCCLUSION_RATIO,
    train_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
    category_aug_config: Path = CATEGORY_AUGMENTATION_CONFIG_PATH,
    target_samples: Optional[int] = None,
    max_aug: int = 10,
    robbery_class: int = ROBBERY_CLASS,
    min_robbery_rows: Optional[int] = None,
    min_negative_rows: Optional[int] = None,
    negative_to_robbery_ratio: float = PREFLIGHT_NEGATIVE_TO_ROBBERY_RATIO,
    write_config: bool = False,
    output_config: Optional[Path] = None,
) -> Dict[str, Any]:
    header("1) Recolección de ejemplos (mismo criterio que train_model_operations)")
    examples = collect_examples(
        pose_source=pose_source,
        single_user_only=single_user_only,
        min_clip_seconds=float(min_clip_seconds),
        min_valid_frames=int(min_valid_frames),
        min_valid_pct=float(min_valid_pct),
        max_occlusion_ratio=float(max_occlusion_ratio),
    )
    n_total = len(examples)
    global_counts = count_examples_by_action_label(examples)
    print(f"  Total ejemplos: {CYAN}{n_total}{RESET} | categorías: {len(global_counts)}")

    header("2) Split train / val / test")
    if (train_ratio is None) ^ (val_ratio is None):
        raise ValueError("Indica ambos --train-ratio y --val-ratio, o ninguno para heurística automática.")
    if train_ratio is None or val_ratio is None:
        tr, vr, te = suggest_split_ratios(n_total)
        print(f"  Heurística suggest_split_ratios(N={n_total}): train={tr:.3f} val={vr:.3f} test={te:.3f}")
    else:
        tr, vr = float(train_ratio), float(val_ratio)
        if tr + vr > 1.0:
            raise ValueError(f"train_ratio+val_ratio debe ser <= 1.0 (got {tr}+{vr})")
        te = 1.0 - tr - vr
        print(f"  Ratios explícitos: train={tr:.3f} val={vr:.3f} test={te:.3f}")

    train_ex, val_ex, test_ex = split_examples(examples, seed=SEED, train_ratio=tr, val_ratio=vr)
    train_counts = count_examples_by_action_label(train_ex)
    val_counts = count_examples_by_action_label(val_ex)
    test_counts = count_examples_by_action_label(test_ex)

    print(
        f"  Clips => train: {GREEN}{len(train_ex)}{RESET} ({_pct(len(train_ex), n_total):.1f}%) | "
        f"val: {len(val_ex)} ({_pct(len(val_ex), n_total):.1f}%) | "
        f"test: {len(test_ex)} ({_pct(len(test_ex), n_total):.1f}%)"
    )
    print(f"  {YELLOW}Val/test: sin data augmentation en entrenamiento.{RESET}")

    header("3) Augmentación por categoría (solo train)")
    cfg_path = Path(category_aug_config)
    current_cfg = load_category_augmentation_config(cfg_path)
    include_identity = bool(current_cfg.get("include_identity", True))
    print(f"  Config actual: {cfg_path} | activa={_category_augment_is_active(current_cfg)}")

    current_aug = _aug_map_for_train(train_counts, current_cfg)
    proposed_aug = propose_category_augment_counts(
        train_counts,
        robbery_class=robbery_class,
        target_samples=target_samples,
        min_robbery_rows=min_robbery_rows,
        min_negative_rows=min_negative_rows,
        negative_to_robbery_ratio=negative_to_robbery_ratio,
        max_aug=max_aug,
        include_identity=include_identity,
    )
    for cat in train_counts:
        proposed_aug.setdefault(int(cat), 0)

    balance = analyze_robbery_augment_balance(
        train_counts,
        proposed_aug,
        robbery_class=robbery_class,
        include_identity=include_identity,
        negative_to_robbery_ratio=negative_to_robbery_ratio,
    )
    min_rob_used = int(min_robbery_rows or PREFLIGHT_MIN_ROBBERY_TRAIN_ROWS)
    min_neg_used = int(min_negative_rows or PREFLIGHT_MIN_NEGATIVE_TRAIN_ROWS)
    _print_robbery_balance(
        balance,
        robbery_class=robbery_class,
        min_robbery_rows=min_rob_used,
        min_negative_rows=min_neg_used,
        neg_ratio_target=negative_to_robbery_ratio,
    )

    proposed_cfg = _build_proposed_config(
        proposed_aug,
        include_identity=include_identity,
        default=int(current_cfg.get("default", 0) or 0),
        robbery_class=robbery_class,
    )
    current_rows = summarize_category_aug_on_train(train_counts, current_cfg)
    proposed_rows = summarize_category_aug_on_train(train_counts, proposed_cfg)

    if target_samples is None:
        min_rob_used = int(min_robbery_rows or max(PREFLIGHT_MIN_ROBBERY_TRAIN_ROWS, 40))
        print(
            f"  Objetivos augment: robo (clase {robbery_class}) ≥ ~{min_rob_used} filas | "
            f"negativos ≥ ~{min_neg_used}/clase | ratio neg/robo ≥ {negative_to_robbery_ratio:.1f} | "
            f"max_aug={max_aug}"
        )
    else:
        print(
            f"  target_samples={target_samples} | ratio neg/robo ≥ {negative_to_robbery_ratio:.1f} | "
            f"max_aug={max_aug}"
        )

    _print_cat_table(
        global_counts,
        train_counts,
        val_counts,
        test_counts,
        current_aug,
        proposed_aug,
        current_rows,
        proposed_rows,
    )

    header("6) Resumen general antes de entrenar")
    print(f"  Pool original total:     {n_total} clips")
    print(f"  Train (sin augment):     {len(train_ex)} clips")
    print(f"  Train (config actual):   {sum(v['train_rows'] for v in current_rows.values())} filas dataset")
    print(f"  Train (config propuesta): {sum(v['train_rows'] for v in proposed_rows.values())} filas dataset")
    print(f"  Val (sin augment):       {len(val_ex)} clips")
    print(f"  Test (sin augment):      {len(test_ex)} clips")
    print(
        f"\n  {GREEN}Recomendación:{RESET} revisa la columna aug_prop y ejecuta:\n"
        f"    python preflight_train.py --write-config\n"
        f"  o copia manualmente los valores a {cfg_path.name}"
    )

    out_path = Path(output_config or cfg_path)
    if write_config:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(proposed_cfg, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"\n  {GREEN}[OK]{RESET} Config propuesta escrita en: {out_path}")

    return {
        "total": n_total,
        "split_ratios": {"train": tr, "val": vr, "test": te},
        "train_clips": len(train_ex),
        "val_clips": len(val_ex),
        "test_clips": len(test_ex),
        "global_by_category": global_counts,
        "train_by_category": train_counts,
        "proposed_aug": proposed_aug,
        "robbery_balance": balance,
        "proposed_config": proposed_cfg,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight de entrenamiento: clips, split y augment por categoría.")
    parser.add_argument("--pose-source", choices=["filtered", "full"], default="filtered")
    parser.add_argument("--single-user-only", action="store_true")
    parser.add_argument("--min-clip-seconds", type=float, default=MIN_CLIP_SECONDS)
    parser.add_argument("--min-valid-frames", type=int, default=MIN_VALID_FRAMES)
    parser.add_argument("--min-valid-pct", type=float, default=MIN_VALID_PCT)
    parser.add_argument("--max-occlusion-ratio", type=float, default=MAX_OCCLUSION_RATIO)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument(
        "--category-aug-config",
        type=str,
        default=str(CATEGORY_AUGMENTATION_CONFIG_PATH),
        help="JSON de augment por categoría (lectura y destino por defecto con --write-config).",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=None,
        help="Filas efectivas objetivo por categoría en train (default: mediana del train, cap 20–150).",
    )
    parser.add_argument("--max-aug", type=int, default=10, help="Máximo de variantes augmentadas por clip.")
    parser.add_argument(
        "--robbery-class",
        type=int,
        default=ROBBERY_CLASS,
        help=f"Clase de robo (default {ROBBERY_CLASS}).",
    )
    parser.add_argument(
        "--min-robbery-rows",
        type=int,
        default=None,
        help=f"Filas efectivas mínimas en train para robo/recall (default {PREFLIGHT_MIN_ROBBERY_TRAIN_ROWS}).",
    )
    parser.add_argument(
        "--min-negative-rows",
        type=int,
        default=None,
        help=f"Filas efectivas mínimas por clase no-robo (default {PREFLIGHT_MIN_NEGATIVE_TRAIN_ROWS}).",
    )
    parser.add_argument(
        "--negative-to-robbery-ratio",
        type=float,
        default=PREFLIGHT_NEGATIVE_TO_ROBBERY_RATIO,
        help=(
            "Ratio total filas negativas / filas robo en train tras augment "
            f"(default {PREFLIGHT_NEGATIVE_TO_ROBBERY_RATIO}, reduce FP)."
        ),
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Escribe la config propuesta en --category-aug-config (sobrescribe).",
    )
    parser.add_argument(
        "--output-config",
        type=str,
        default=None,
        help="Ruta alternativa al escribir (--write-config); si no se indica, usa --category-aug-config.",
    )
    args = parser.parse_args()

    try:
        run_preflight(
            pose_source=args.pose_source,
            single_user_only=args.single_user_only,
            min_clip_seconds=args.min_clip_seconds,
            min_valid_frames=args.min_valid_frames,
            min_valid_pct=args.min_valid_pct,
            max_occlusion_ratio=args.max_occlusion_ratio,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            category_aug_config=Path(args.category_aug_config),
            target_samples=args.target_samples,
            max_aug=args.max_aug,
            robbery_class=args.robbery_class,
            min_robbery_rows=args.min_robbery_rows,
            min_negative_rows=args.min_negative_rows,
            negative_to_robbery_ratio=args.negative_to_robbery_ratio,
            write_config=args.write_config,
            output_config=Path(args.output_config) if args.output_config else None,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"{YELLOW}[ERROR]{RESET} {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
