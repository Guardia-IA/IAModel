#!/usr/bin/env python3
"""
Genera planes y configs de augment para cada celda de la campaña.

Uso:
  cd experiments/training/campaign
  python preflight_campaign.py --write-all
  python preflight_campaign.py --cells bin_full bin_filtered --write-all
  python preflight_campaign.py --write-all --skip-time-estimate
"""
from __future__ import annotations

import argparse
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
        load_campaign_config,
        filter_cells,
        ensure_cell_dirs,
        training_plan_path,
        category_aug_path,
        class_map_path,
    )
    from class_map_utils import load_class_map, adjust_augment_for_fp_hardened
    from preflight_train_plan import build_training_plan, write_training_plan, header, BOLD, RESET, CYAN, YELLOW
    from model_config import ROBBERY_CLASS, PREFLIGHT_NEGATIVE_TO_ROBBERY_RATIO
    from training_time_estimate import fmt_duration
except ImportError as exc:
    raise SystemExit(
        "Ejecuta desde experiments/training/campaign con el entorno que tenga torch.\n"
        f"Import error: {exc}"
    ) from exc


def _aug_profile(config: Dict[str, Any], profile_id: str) -> Dict[str, Any]:
    profiles = config.get("aug_profiles") or {}
    if profile_id not in profiles:
        raise KeyError(f"aug_profile desconocido: {profile_id!r}")
    return profiles[profile_id]


def run_preflight_cell(
    cell: Dict[str, Any],
    config: Dict[str, Any],
    *,
    data_root: Optional[Path] = None,
    write: bool = False,
    skip_time_estimate: bool = False,
    experiment_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    cell_id = cell["id"]
    arts = ensure_cell_dirs(cell_id)
    aug_prof = _aug_profile(config, cell["aug_profile"])
    class_map_spec = load_class_map(class_map_path(cell["class_map_id"]))
    exp_ids = experiment_ids or list(config.get("experiment_ids", []))

    neg_ratio = float(aug_prof.get("negative_to_robbery_ratio", PREFLIGHT_NEGATIVE_TO_ROBBERY_RATIO))
    plan = build_training_plan(
        task=cell["task"],
        positive_class=int(config.get("robbery_class", ROBBERY_CLASS)),
        pose_source=cell["pose_source"],
        single_user_only=bool(config.get("single_user_only", True)),
        data_root=data_root,
        category_aug_config=category_aug_path(cell_id),
        negative_to_robbery_ratio=neg_ratio,
        skip_time_estimate=skip_time_estimate,
        class_map_spec=class_map_spec,
        experiment_ids=exp_ids if exp_ids else None,
    )

    plan["campaign"] = {
        "cell_id": cell_id,
        "class_map_id": cell["class_map_id"],
        "aug_profile": cell["aug_profile"],
        "experiment_ids": exp_ids,
    }

    proposed = plan.get("proposed_category_augmentation") or {}
    cats = proposed.get("categories") or {}
    proposed_counts = {int(k): int(v) for k, v in cats.items()}

    if aug_prof.get("fp_hardened"):
        proposed_counts = adjust_augment_for_fp_hardened(
            proposed_counts,
            robbery_class=int(config.get("robbery_class", ROBBERY_CLASS)),
        )
        proposed["categories"] = {str(k): int(v) for k, v in sorted(proposed_counts.items())}
        plan["proposed_category_augmentation"] = proposed
        plan["campaign"]["aug_fp_hardened"] = True

    time_est = plan.get("training_time_estimate") or {}
    cell_seconds = float(time_est.get("primary_total_seconds", 0.0))

    if write:
        cfg_out = category_aug_path(cell_id)
        cfg_out.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_out, "w", encoding="utf-8") as f:
            json.dump(proposed, f, indent=2, ensure_ascii=False)
            f.write("\n")
        plan["category_augmentation_config"] = str(cfg_out.resolve())

        plan_out = training_plan_path(cell_id)
        write_training_plan(plan, plan_out)
        print(f"  [OK] {cell_id}: plan → {plan_out}")
        print(f"       augment → {cfg_out}")

    return {
        "cell_id": cell_id,
        "plan_path": str(training_plan_path(cell_id)),
        "aug_path": str(category_aug_path(cell_id)),
        "task": cell["task"],
        "pose_source": cell["pose_source"],
        "class_map_id": cell["class_map_id"],
        "aug_profile": cell["aug_profile"],
        "train_rows": plan.get("totals", {}).get("rows_train_proposed"),
        "models_dir": str(arts["models_dir"]),
        "time_estimate_seconds": cell_seconds,
        "time_estimate_human": time_est.get("primary_total_human"),
        "experiments_count": len(exp_ids),
    }


def print_campaign_time_rollup(summary: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
    rows = [r for r in summary if "time_estimate_seconds" in r and "error" not in r]
    if not rows:
        return
    total_s = sum(float(r.get("time_estimate_seconds", 0.0)) for r in rows)
    n_cells = len(rows)
    n_exp = int(rows[0].get("experiments_count", 0)) if rows else 0
    total_runs = n_cells * n_exp

    header("Resumen estimación — campaña completa")
    print(f"  Celdas: {CYAN}{n_cells}{RESET} | experimentos/celda: {n_exp} | entrenamientos totales: {total_runs}")
    print(f"  Tiempo estimado acumulado (GPU/CPU según tu máquina): {BOLD}{fmt_duration(total_s)}{RESET}")
    print(f"  ({total_s / 3600.0:.1f} h)")
    print(f"\n  {'Celda':<22} | {'Train rows':>10} | {'Tiempo':>12}")
    print("  " + "-" * 50)
    for r in rows:
        hum = r.get("time_estimate_human") or fmt_duration(float(r.get("time_estimate_seconds", 0)))
        print(
            f"  {r['cell_id']:<22} | {r.get('train_rows', 0):>10} | {hum:>12}"
        )
    print(f"{RESET}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Preflight de la campaña (planes por celda)")
    ap.add_argument("--config", type=str, default=None, help="campaign_config.json")
    ap.add_argument("--cells", nargs="*", default=None, help="IDs de celdas (default: todas)")
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--write-all", action="store_true", help="Escribe plan + augment por celda")
    ap.add_argument(
        "--skip-time-estimate",
        action="store_true",
        help="Omite la estimación de tiempo (más rápido, sin sección 7).",
    )
    ap.add_argument(
        "--skip-validate",
        action="store_true",
        help="No ejecuta validate_campaign al final.",
    )
    args = ap.parse_args()

    config = load_campaign_config(Path(args.config) if args.config else None)
    cells = filter_cells(config, args.cells)
    if not cells:
        print("No hay celdas que procesar.", file=sys.stderr)
        return 1

    data_root = Path(args.data_root) if args.data_root else None
    exp_ids = list(config.get("experiment_ids", []))
    print(f"\n=== Preflight campaña — {len(cells)} celdas ===")
    if exp_ids and not args.skip_time_estimate:
        print(f"Estimación por celda: experimentos {exp_ids}\n")
    summary: List[Dict[str, Any]] = []
    for cell in cells:
        print(f"\n--- Celda: {cell['id']} ({cell['task']}, {cell['pose_source']}) ---")
        try:
            row = run_preflight_cell(
                cell,
                config,
                data_root=data_root,
                write=args.write_all,
                skip_time_estimate=args.skip_time_estimate,
                experiment_ids=exp_ids,
            )
            summary.append(row)
        except Exception as exc:
            print(f"  ERROR {cell['id']}: {exc}", file=sys.stderr)
            summary.append({"cell_id": cell["id"], "error": str(exc)})

    if not args.skip_time_estimate:
        print_campaign_time_rollup(summary, config)

    if not getattr(args, "skip_validate", False):
        try:
            from validate_campaign import run_validation, print_report

            print(f"\n{BOLD}=== Validación post-preflight ==={RESET}\n")
            vreport = run_validation(
                config_path=Path(args.config) if args.config else None,
                data_root=data_root,
                require_plans=bool(args.write_all),
            )
            print_report(vreport)
            if not vreport.passed:
                return 1
        except ImportError as exc:
            print(f"  {YELLOW}[!] validate_campaign no disponible: {exc}{RESET}")

    master = CAMPAIGN_DIR / "artifacts" / "preflight_summary.json"
    master.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_ids": exp_ids,
        "cells": summary,
        "campaign_time_estimate_seconds": sum(
            float(r.get("time_estimate_seconds", 0.0))
            for r in summary
            if "time_estimate_seconds" in r
        ),
    }
    with open(master, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\nResumen: {master}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
