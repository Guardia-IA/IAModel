#!/usr/bin/env python3
"""
Preflight para entrenamiento con augmentación masiva (~100k filas train).

Genera training_plan.json + config_mass_augmentation.json por celda,
estima tiempo con filas expandidas y valida splits (val/test solo clips reales).

Uso:
  python preflight_mass_augment.py --write-all --cells mc_full bin_full
  python preflight_mass_augment.py --write-all --run-id mass_20260707_120000
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

CAMPAIGN_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_DIR))

try:
    from campaign_paths import (
        load_merged_campaign_config,
        filter_cells,
        artifacts_root,
        training_plan_path,
        category_aug_path,
        mass_augment_config_path,
        class_map_path,
        ensure_cell_dirs,
        resolve_experiment_ids,
        CONFIG_PATH,
    )
    from preflight_campaign import _resolve_cell_settings, print_campaign_time_rollup
    from preflight_train_plan import (
        build_training_plan,
        write_training_plan,
        header,
        CYAN,
        YELLOW,
        RESET,
    )
    from class_map_utils import load_class_map
    from model_config import ROBBERY_CLASS, PREFLIGHT_NEGATIVE_TO_ROBBERY_RATIO
    from mass_augment import (
        load_mass_augment_config,
        compute_mass_augment_plan,
        mass_aug_to_category_config,
        apply_mass_plan_to_training_plan,
    )
    from training_time_estimate import estimate_all_experiments, fmt_duration
except ImportError as exc:
    raise SystemExit(f"Import error: {exc}") from exc


DEFAULT_MASS_CELLS = ("mc_full", "mc_filtered", "bin_full", "bin_filtered")


def resolve_mass_cells(config: Dict[str, Any], cell_ids: Optional[List[str]]) -> List[Dict[str, Any]]:
    ma = config.get("mass_augment") or {}
    default_ids = ma.get("cells") or list(DEFAULT_MASS_CELLS)
    ids = cell_ids or default_ids
    return filter_cells(config, list(ids))


def load_mass_config(config: Dict[str, Any], override: Optional[Path] = None) -> Dict[str, Any]:
    ma = config.get("mass_augment") or {}
    rel = str(ma.get("config_path", "config_mass_augmentation.json"))
    path = override or (TRAINING_DIR / rel)
    if not path.is_file():
        path = Path(rel)
    cfg = load_mass_augment_config(path)
    for key in ("variants_per_clip", "target_train_rows", "include_identity", "max_extra_variants_per_clip", "cap_at_target"):
        if key in ma and ma[key] is not None:
            cfg[key] = ma[key]
    if ma.get("synthetic_eval"):
        cfg["synthetic_eval"] = {**(cfg.get("synthetic_eval") or {}), **ma["synthetic_eval"]}
    return cfg


def _build_base_plan(
    cell: Dict[str, Any],
    config: Dict[str, Any],
    *,
    data_root: Optional[Path],
    experiment_ids: Optional[List[int]],
    run_id: Optional[str],
) -> tuple[Dict[str, Any], List[int], str]:
    cell_id = cell["id"]
    ensure_cell_dirs(cell_id, run_id=run_id)
    aug_prof, aug_profile_id, _improve_meta = _resolve_cell_settings(cell, config)
    class_map_spec = load_class_map(class_map_path(cell["class_map_id"]))
    if experiment_ids is not None:
        exp_ids = resolve_experiment_ids(experiment_ids)
    else:
        exp_ids = resolve_experiment_ids(config.get("experiment_ids") or "all")

    neg_ratio = float(aug_prof.get("negative_to_robbery_ratio", PREFLIGHT_NEGATIVE_TO_ROBBERY_RATIO))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        plan = build_training_plan(
            task=cell["task"],
            positive_class=int(config.get("robbery_class", ROBBERY_CLASS)),
            pose_source=cell["pose_source"],
            single_user_only=bool(config.get("single_user_only", True)),
            data_root=data_root,
            category_aug_config=category_aug_path(cell_id, run_id=run_id),
            negative_to_robbery_ratio=neg_ratio,
            skip_time_estimate=True,
            class_map_spec=class_map_spec,
            experiment_ids=exp_ids if exp_ids else None,
        )
    plan["campaign"] = {
        "cell_id": cell_id,
        "class_map_id": cell["class_map_id"],
        "aug_profile": aug_profile_id,
        "experiment_ids": exp_ids,
        "mass_augment_run": True,
    }
    if run_id:
        plan["campaign"]["run_id"] = run_id
    return plan, exp_ids, aug_profile_id, buf.getvalue()


def run_mass_preflight_cell(
    cell: Dict[str, Any],
    config: Dict[str, Any],
    mass_cfg: Dict[str, Any],
    *,
    data_root: Optional[Path] = None,
    write: bool = False,
    skip_time_estimate: bool = False,
    experiment_ids: Optional[List[int]] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    cell_id = cell["id"]
    plan, exp_ids, aug_profile_id, _base_log = _build_base_plan(
        cell,
        config,
        data_root=data_root,
        experiment_ids=experiment_ids,
        run_id=run_id,
    )

    std_totals = (plan.get("dataset_stats") or {}).get("totals") or {}
    std_synthetic = int(std_totals.get("rows_synthetic_train", 0))
    std_real_train = int(std_totals.get("clips_real_train", 0))

    train_counts = {
        int(k): int(v)
        for k, v in (plan.get("split_stats_by_category", {}).get("train") or {}).items()
    }
    mass_plan = compute_mass_augment_plan(train_counts, mass_cfg)
    cat_cfg = mass_aug_to_category_config(mass_cfg, mass_plan)
    apply_mass_plan_to_training_plan(plan, mass_plan, cat_cfg)

    plan_path = training_plan_path(cell_id, run_id=run_id)
    mass_cfg_path = mass_augment_config_path(cell_id, run_id=run_id)
    cat_cfg_path = category_aug_path(cell_id, run_id=run_id)

    plan["mass_augmentation"] = {
        "enabled": True,
        "config_path": str(mass_cfg_path.resolve()),
        "plan": mass_plan,
        "recipes_count": len(mass_cfg.get("recipes") or []),
    }
    plan["proposed_category_augmentation"] = cat_cfg
    plan["category_augmentation_config"] = str(cat_cfg_path.resolve())

    ds_totals = (plan.get("dataset_stats") or {}).get("totals") or {}
    projected_train = int(ds_totals.get("rows_total_train", mass_plan.get("projected_total_rows", 0)))
    synthetic_train = int(ds_totals.get("rows_synthetic_train", 0))
    real_train = int(mass_plan.get("train_clips", std_real_train))
    val_rows = int(ds_totals.get("clips_real_val", 0))
    cell_seconds = 0.0
    time_human: Optional[str] = None

    if not skip_time_estimate and exp_ids:
        time_summary = estimate_all_experiments(
            train_rows=projected_train,
            val_rows=val_rows,
            experiment_ids=exp_ids,
        )
        cell_seconds = float(time_summary["primary_total_seconds"])
        time_human = fmt_duration(cell_seconds)
        plan["training_time_estimate"] = {
            "train_rows_per_epoch": projected_train,
            "val_rows_per_epoch": val_rows,
            "experiments_pending": time_summary["experiments_pending"],
            "experiments_skipped_done": time_summary["experiments_skipped_done"],
            "cuda_available": time_summary["cuda_available"],
            "total_gpu_seconds": time_summary["total_gpu_seconds"],
            "total_cpu_seconds": time_summary["total_cpu_seconds"],
            "primary_device": time_summary["primary_device"],
            "primary_total_seconds": cell_seconds,
            "primary_total_human": time_human,
            "mass_augment_projected_train_rows": projected_train,
        }

    header(f"Mass augment — {cell_id}")
    print(f"  Clips train (UIDs reales en split): {real_train}")
    print(f"  Variantes por clip: {mass_plan.get('variants_per_clip')} (+ extras balanceo)")
    print(
        f"  {CYAN}TRAIN MASIVO: {real_train} clips × ~{mass_plan.get('variants_per_clip')} "
        f"= {projected_train} filas augmentadas{RESET}"
    )
    print(
        f"  (Augmentación estándar del preflight: {std_synthetic} sintéticas — "
        f"{YELLOW}no aplica{RESET}; se usa mass_augmentation)"
    )
    print(f"  Val/test: solo clips reales ({val_rows} val)")
    if time_human:
        print(f"  Tiempo estimado ({len(exp_ids)} exp): {YELLOW}{time_human}{RESET}")

    if write:
        with open(mass_cfg_path, "w", encoding="utf-8") as f:
            json.dump(dict(mass_cfg), f, indent=2, ensure_ascii=False)
            f.write("\n")
        with open(cat_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cat_cfg, f, indent=2, ensure_ascii=False)
            f.write("\n")
        write_training_plan(plan, plan_path)
        print(f"  [OK] plan → {plan_path}")
        print(f"       mass aug → {mass_cfg_path}")

    return {
        "cell_id": cell_id,
        "run_id": run_id,
        "plan_path": str(plan_path),
        "aug_path": str(cat_cfg_path),
        "mass_aug_path": str(mass_cfg_path),
        "task": cell["task"],
        "pose_source": cell["pose_source"],
        "class_map_id": cell["class_map_id"],
        "aug_profile": aug_profile_id,
        "train_rows": projected_train,
        "train_rows_projected": projected_train,
        "train_clips_real": real_train,
        "train_rows_synthetic": synthetic_train,
        "mass_augment": mass_plan,
        "experiments_count": len(exp_ids),
        "time_estimate_seconds": cell_seconds,
        "time_estimate_human": time_human,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Preflight augmentación masiva")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--write-all", action="store_true")
    ap.add_argument("--skip-time-estimate", action="store_true")
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--mass-config", type=str, default=None)
    ap.add_argument("--experiment-ids", nargs="*", default=None)
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config) if args.config else CONFIG_PATH
    config = load_merged_campaign_config(config_path)
    cells = resolve_mass_cells(config, args.cells)
    if not cells:
        print("No hay celdas.", file=sys.stderr)
        return 1

    data_root = Path(args.data_root).expanduser() if args.data_root else None
    run_id = str(args.run_id).strip() if args.run_id else None
    mass_cfg = load_mass_config(config, Path(args.mass_config) if args.mass_config else None)
    exp_ids = resolve_experiment_ids(args.experiment_ids or config.get("experiment_ids") or "all")

    print(f"\n=== Preflight MASS AUG — {len(cells)} celdas ===")
    if run_id:
        print(f"Run ID: {run_id} → {artifacts_root(run_id)}")

    summary: List[Dict[str, Any]] = []
    for cell in cells:
        print(f"\n--- Celda: {cell['id']} ({cell['task']}, {cell['pose_source']}) ---")
        try:
            summary.append(
                run_mass_preflight_cell(
                    cell,
                    config,
                    mass_cfg,
                    data_root=data_root,
                    write=args.write_all,
                    skip_time_estimate=args.skip_time_estimate,
                    experiment_ids=exp_ids,
                    run_id=run_id,
                )
            )
        except Exception as exc:
            print(f"  ERROR {cell['id']}: {exc}", file=sys.stderr)
            summary.append({"cell_id": cell["id"], "error": str(exc)})

    if not args.skip_time_estimate:
        print_campaign_time_rollup(summary, config, run_id=run_id)

    if not args.skip_validate and args.write_all:
        from validate_campaign import run_validation, print_report

        print(f"\n=== Validación post-preflight (mass aug) ===\n")
        vreport = run_validation(
            config_path=config_path,
            data_root=data_root,
            require_plans=True,
            run_id=run_id,
            cell_ids=[c["id"] for c in cells],
        )
        print_report(vreport)
        if not vreport.passed:
            return 1

    master = artifacts_root(run_id) / "preflight_mass_augment_summary.json"
    master.parent.mkdir(parents=True, exist_ok=True)
    with open(master, "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "cells": summary}, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\nResumen: {master}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
