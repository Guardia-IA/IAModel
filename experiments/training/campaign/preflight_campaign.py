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
        load_merged_campaign_config,
        filter_cells,
        ensure_cell_dirs,
        training_plan_path,
        category_aug_path,
        hard_negative_manifest_path,
        run_meta_path,
        class_map_path,
        artifacts_root,
        CONFIG_PATH,
        resolve_experiment_ids,
    )
    from improve_utils import (
        load_fp_manifest_csv,
        extract_uids_from_fp_rows,
        apply_uniform_ops_per_clip,
        boost_aug_from_fp_categories,
        write_hard_negative_manifest,
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
    return dict(profiles[profile_id])


def _resolve_cell_settings(
    cell: Dict[str, Any],
    config: Dict[str, Any],
    *,
    improve_profile_override: Optional[str] = None,
    hard_negative_csv: Optional[Path] = None,
    uniform_ops_per_clip: Optional[int] = None,
    fp_category_boost: Optional[float] = None,
) -> tuple[Dict[str, Any], str, Dict[str, Any]]:
    """Devuelve (aug_prof, aug_profile_id, improve_meta)."""
    improve_id = improve_profile_override or cell.get("improve_profile")
    if improve_id:
        profiles = config.get("improve_profiles") or {}
        if improve_id not in profiles:
            raise KeyError(f"improve_profile desconocido: {improve_id!r}")
        ip = profiles[improve_id]
        base_aug_id = str(ip.get("base_aug_profile", "fp_hardened"))
        aug_prof = _aug_profile(config, base_aug_id)
        train_opts = dict(aug_prof.get("train_opts") or {})
        train_opts.update(ip.get("train_opts") or {})
        aug_prof["train_opts"] = train_opts
        meta = {
            "improve_profile": improve_id,
            "base_aug_profile": base_aug_id,
            "hard_negative_csv": str(hard_negative_csv) if hard_negative_csv else ip.get("hard_negative_csv"),
            "hard_negative_uid_weight": float(ip.get("hard_negative_uid_weight", 3.0)),
            "fp_category_boost": float(
                fp_category_boost
                if fp_category_boost is not None
                else ip.get("fp_category_boost", 0.0)
            ),
            "uniform_ops_per_clip": int(
                uniform_ops_per_clip
                if uniform_ops_per_clip is not None
                else ip.get("uniform_ops_per_clip", 0)
            ),
        }
        return aug_prof, base_aug_id, meta

    aug_id = str(cell["aug_profile"])
    return _aug_profile(config, aug_id), aug_id, {}


def run_preflight_cell(
    cell: Dict[str, Any],
    config: Dict[str, Any],
    *,
    data_root: Optional[Path] = None,
    write: bool = False,
    skip_time_estimate: bool = False,
    experiment_ids: Optional[List[int]] = None,
    run_id: Optional[str] = None,
    improve_profile_override: Optional[str] = None,
    hard_negative_csv: Optional[Path] = None,
    uniform_ops_per_clip: Optional[int] = None,
    fp_category_boost: Optional[float] = None,
) -> Dict[str, Any]:
    cell_id = cell["id"]
    arts = ensure_cell_dirs(cell_id, run_id=run_id)
    aug_prof, aug_profile_id, improve_meta = _resolve_cell_settings(
        cell,
        config,
        improve_profile_override=improve_profile_override,
        hard_negative_csv=hard_negative_csv,
        uniform_ops_per_clip=uniform_ops_per_clip,
        fp_category_boost=fp_category_boost,
    )
    if hard_negative_csv and not improve_meta.get("hard_negative_csv"):
        improve_meta["hard_negative_csv"] = str(hard_negative_csv)
        improve_meta.setdefault("hard_negative_uid_weight", 3.0)
        improve_meta.setdefault("fp_category_boost", 1.5)
    class_map_spec = load_class_map(class_map_path(cell["class_map_id"]))
    if experiment_ids is not None:
        exp_ids = resolve_experiment_ids(experiment_ids)
    elif config.get("experiment_ids") is not None:
        exp_ids = resolve_experiment_ids(config.get("experiment_ids"))
    else:
        exp_ids = []

    neg_ratio = float(aug_prof.get("negative_to_robbery_ratio", PREFLIGHT_NEGATIVE_TO_ROBBERY_RATIO))
    plan = build_training_plan(
        task=cell["task"],
        positive_class=int(config.get("robbery_class", ROBBERY_CLASS)),
        pose_source=cell["pose_source"],
        single_user_only=bool(config.get("single_user_only", True)),
        data_root=data_root,
        category_aug_config=category_aug_path(cell_id, run_id=run_id),
        negative_to_robbery_ratio=neg_ratio,
        skip_time_estimate=skip_time_estimate,
        class_map_spec=class_map_spec,
        experiment_ids=exp_ids if exp_ids else None,
    )

    plan["campaign"] = {
        "cell_id": cell_id,
        "class_map_id": cell["class_map_id"],
        "aug_profile": aug_profile_id,
        "experiment_ids": exp_ids,
    }
    if run_id:
        plan["campaign"]["run_id"] = run_id
    if improve_meta:
        plan["campaign"]["improve"] = improve_meta

    proposed = plan.get("proposed_category_augmentation") or {}
    cats = proposed.get("categories") or {}
    proposed_counts = {int(k): int(v) for k, v in cats.items()}

    if aug_prof.get("fp_hardened"):
        proposed_counts = adjust_augment_for_fp_hardened(
            proposed_counts,
            robbery_class=int(config.get("robbery_class", ROBBERY_CLASS)),
        )

    fp_rows: List[Dict[str, str]] = []
    hn_csv_path: Optional[Path] = None
    if improve_meta.get("hard_negative_csv"):
        hn_csv_path = Path(str(improve_meta["hard_negative_csv"]))
        if hn_csv_path.is_file():
            fp_rows = load_fp_manifest_csv(hn_csv_path)
            boost = float(improve_meta.get("fp_category_boost") or 0.0)
            if boost > 0:
                proposed_counts = boost_aug_from_fp_categories(
                    proposed_counts,
                    fp_rows,
                    boost_factor=boost,
                    robbery_class=int(config.get("robbery_class", ROBBERY_CLASS)),
                )

    uniform_ops = int(improve_meta.get("uniform_ops_per_clip") or 0)
    if uniform_ops > 0:
        proposed_counts = apply_uniform_ops_per_clip(
            proposed_counts,
            uniform_ops,
            robbery_class=int(config.get("robbery_class", ROBBERY_CLASS)),
        )

    if aug_prof.get("fp_hardened") or fp_rows or uniform_ops > 0:
        proposed["categories"] = {str(k): int(v) for k, v in sorted(proposed_counts.items())}
        plan["proposed_category_augmentation"] = proposed
        plan["campaign"]["aug_fp_hardened"] = bool(aug_prof.get("fp_hardened"))

    hn_manifest_out: Optional[Path] = None
    if fp_rows and write:
        uids = extract_uids_from_fp_rows(fp_rows)
        if uids:
            hn_manifest_out = write_hard_negative_manifest(
                uids,
                hard_negative_manifest_path(cell_id, run_id=run_id),
                source_csv=hn_csv_path,
                weight=float(improve_meta.get("hard_negative_uid_weight", 3.0)),
            )
            plan["campaign"]["hard_negative_manifest"] = str(hn_manifest_out.resolve())
            print(f"  [HN] {len(uids)} UIDs hard-negative → {hn_manifest_out}")

    time_est = plan.get("training_time_estimate") or {}
    cell_seconds = float(time_est.get("primary_total_seconds", 0.0))

    if write:
        cfg_out = category_aug_path(cell_id, run_id=run_id)
        cfg_out.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_out, "w", encoding="utf-8") as f:
            json.dump(proposed, f, indent=2, ensure_ascii=False)
            f.write("\n")
        plan["category_augmentation_config"] = str(cfg_out.resolve())

        plan_out = training_plan_path(cell_id, run_id=run_id)
        write_training_plan(plan, plan_out)
        print(f"  [OK] {cell_id}: plan → {plan_out}")
        print(f"       augment → {cfg_out}")

    return {
        "cell_id": cell_id,
        "run_id": run_id,
        "plan_path": str(training_plan_path(cell_id, run_id=run_id)),
        "aug_path": str(category_aug_path(cell_id, run_id=run_id)),
        "task": cell["task"],
        "pose_source": cell["pose_source"],
        "class_map_id": cell["class_map_id"],
        "aug_profile": aug_profile_id,
        "improve_profile": improve_meta.get("improve_profile"),
        "hard_negative_manifest": str(hn_manifest_out) if hn_manifest_out else plan.get("campaign", {}).get("hard_negative_manifest"),
        "train_rows": plan.get("totals", {}).get("rows_train_proposed"),
        "models_dir": str(arts["models_dir"]),
        "time_estimate_seconds": cell_seconds,
        "time_estimate_human": time_est.get("primary_total_human"),
        "experiments_count": len(exp_ids),
    }


def build_campaign_time_rollup(
    summary: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    rows = [r for r in summary if "time_estimate_seconds" in r and "error" not in r]
    if not rows:
        return None
    total_s = sum(float(r.get("time_estimate_seconds", 0.0)) for r in rows)
    n_cells = len(rows)
    n_exp = max(int(r.get("experiments_count", 0)) for r in rows)
    total_runs = n_cells * n_exp
    device = "gpu"
    for r in rows:
        plan_path = Path(str(r.get("plan_path", "")))
        if plan_path.is_file():
            try:
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                te = plan.get("training_time_estimate") or {}
                device = str(te.get("primary_device") or device)
                break
            except (OSError, json.JSONDecodeError):
                pass
    return {
        "cells": n_cells,
        "experiments_per_cell": n_exp,
        "total_train_runs": total_runs,
        "total_seconds": total_s,
        "total_human": fmt_duration(total_s),
        "total_hours": round(total_s / 3600.0, 2),
        "primary_device": device,
        "per_cell": [
            {
                "cell_id": r["cell_id"],
                "train_rows": r.get("train_rows"),
                "seconds": float(r.get("time_estimate_seconds", 0.0)),
                "human": r.get("time_estimate_human")
                or fmt_duration(float(r.get("time_estimate_seconds", 0.0))),
            }
            for r in rows
        ],
    }


def print_campaign_time_rollup(
    summary: List[Dict[str, Any]],
    config: Dict[str, Any],
    *,
    run_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    rollup = build_campaign_time_rollup(summary, config)
    if rollup is None:
        print(f"\n{YELLOW}[!] Sin estimación de tiempo (celdas con error o --skip-time-estimate).{RESET}")
        return None

    total_s = float(rollup["total_seconds"])
    n_cells = int(rollup["cells"])
    n_exp = int(rollup["experiments_per_cell"])
    total_runs = int(rollup["total_train_runs"])
    device = rollup.get("primary_device", "gpu")

    lines = [
        "",
        "=" * 72,
        "RESUMEN TOTAL — ESTIMACIÓN DE ENTRENAMIENTO (campaña completa)",
        "=" * 72,
        f"Celdas: {n_cells} | experimentos/celda: {n_exp} | entrenamientos totales: {total_runs}",
        f"Dispositivo de referencia: {device.upper()}",
        f"TIEMPO TOTAL ESTIMADO: {rollup['total_human']} ({rollup['total_hours']} h)",
        "",
        f"{'Celda':<22} | {'Train rows':>10} | {'Tiempo':>12}",
        "-" * 50,
    ]
    for row in rollup["per_cell"]:
        lines.append(
            f"{row['cell_id']:<22} | {row.get('train_rows', 0):>10} | {row['human']:>12}"
        )
    lines.extend(
        [
            "-" * 50,
            f"SUMA TOTAL: {rollup['total_human']} ({rollup['total_hours']} h)",
            "Nota: heurística por arquitectura/epochs; eval no incluida.",
            "=" * 72,
            "",
        ]
    )
    text = "\n".join(lines)

    header("Resumen estimación — campaña completa")
    print(f"  Celdas: {CYAN}{n_cells}{RESET} | experimentos/celda: {n_exp} | entrenamientos: {total_runs}")
    print(f"  Dispositivo ref.: {CYAN}{str(device).upper()}{RESET}")
    print(
        f"  {BOLD}TIEMPO TOTAL ESTIMADO:{RESET} "
        f"{BOLD}{CYAN}{rollup['total_human']}{RESET} ({rollup['total_hours']} h)"
    )
    print(f"\n  {'Celda':<22} | {'Train rows':>10} | {'Tiempo':>12}")
    print("  " + "-" * 50)
    for row in rollup["per_cell"]:
        print(
            f"  {row['cell_id']:<22} | {row.get('train_rows', 0):>10} | {row['human']:>12}"
        )
    print("  " + "-" * 50)
    print(f"  {BOLD}SUMA TOTAL:{RESET} {CYAN}{rollup['total_human']}{RESET} ({rollup['total_hours']} h)")
    print(f"  {YELLOW}Nota: heurística train only; eval no incluida.{RESET}\n")

    if run_id:
        log_dir = artifacts_root(run_id) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        est_log = log_dir / "preflight_time_estimate.txt"
        est_log.write_text(text + "\n", encoding="utf-8")
        print(f"  Estimación guardada en: {CYAN}{est_log}{RESET}\n")

    return rollup


def main() -> int:
    ap = argparse.ArgumentParser(description="Preflight de la campaña (planes por celda)")
    ap.add_argument("--config", type=str, default=None, help="campaign_config.json o campaign_config_improve.json")
    ap.add_argument("--cells", nargs="*", default=None, help="IDs de celdas (default: todas)")
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--write-all", action="store_true", help="Escribe plan + augment por celda")
    ap.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="ID de run (artefactos en artifacts/runs/<run-id>/; no machaca campaña base)",
    )
    ap.add_argument(
        "--improve-profile",
        type=str,
        default=None,
        help="Override del improve_profile de la celda (solo configs improve)",
    )
    ap.add_argument(
        "--hard-negative-csv",
        type=str,
        default=None,
        help="CSV de FP (p. ej. export_ensemble_fp.py) para boost augment + sampler UID",
    )
    ap.add_argument(
        "--uniform-ops-per-clip",
        type=int,
        default=None,
        help="Experimento: mismo N variantes augment por categoría (0=desactivado)",
    )
    ap.add_argument(
        "--fp-category-boost",
        type=float,
        default=None,
        help="Factor extra de augment en categorías del CSV FP",
    )
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
    ap.add_argument(
        "--learning-curve",
        "--prediction",
        dest="learning_curve",
        action="store_true",
        help="Curva de aprendizaje: split maestro + planes por tamaño de train (val/test fijos)",
    )
    ap.add_argument(
        "--train-sizes",
        nargs="+",
        default=None,
        metavar="N|max",
        help="Tamaños train en clips reales (p. ej. 3000 6500 max). 'max' = todos los clips train del split",
    )
    args = ap.parse_args()

    config_path = Path(args.config) if args.config else None
    config = load_merged_campaign_config(config_path)
    data_root = Path(args.data_root) if args.data_root else None
    exp_ids = (
        resolve_experiment_ids(config.get("experiment_ids"))
        if config.get("experiment_ids") is not None
        else []
    )
    run_id = str(args.run_id).strip() if args.run_id else None
    hn_csv = Path(args.hard_negative_csv) if args.hard_negative_csv else None

    if args.learning_curve:
        from learning_curve_utils import (
            parse_train_size_specs,
            resolve_learning_curve_cells,
            run_learning_curve_preflight_all,
        )

        try:
            train_specs = parse_train_size_specs(args.train_sizes, config)
            cells = resolve_learning_curve_cells(config, args.cells)
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        if run_id:
            print("[!] --run-id ignorado en modo --learning-curve (usa lc_<N> y _lc_master)", file=sys.stderr)
        print(f"\n=== Preflight learning curve — {len(cells)} celdas — specs {train_specs} ===")
        summary: List[Dict[str, Any]] = []
        train_sizes: List[int] = []
        try:
            row = run_learning_curve_preflight_all(
                cells,
                config,
                train_size_specs=train_specs,
                data_root=data_root,
                write=args.write_all,
                skip_time_estimate=args.skip_time_estimate,
            )
            summary.append(row)
            train_sizes = row.get("train_sizes") or []
            cell_id = cells[0]["id"]
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            summary.append({"error": str(exc)})
            cell_id = cells[0]["id"] if cells else "?"

        if not args.skip_validate and args.write_all and summary and "error" not in summary[0]:
            try:
                from validate_campaign import run_validation, print_report

                print(f"\n{BOLD}=== Validación post-preflight (learning curve) ==={RESET}\n")
                cell_ids_lc = summary[0].get("cell_ids") or []
                for rid in train_sizes:
                    run_rid = f"lc_{rid}"
                    vreport = run_validation(
                        config_path=config_path,
                        data_root=data_root,
                        require_plans=True,
                        run_id=run_rid,
                        cell_ids=cell_ids_lc,
                    )
                    print_report(vreport)
                    if not vreport.passed:
                        return 1
            except ImportError as exc:
                print(f"  {YELLOW}[!] validate_campaign no disponible: {exc}{RESET}")

        master_root = artifacts_root("_lc_master")
        master_root.mkdir(parents=True, exist_ok=True)
        master = master_root / "preflight_learning_curve_summary.json"
        with open(master, "w", encoding="utf-8") as f:
            json.dump({"train_sizes": train_sizes, "cells": summary}, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"\nResumen learning curve: {master}")
        return 0 if summary and "error" not in summary[0] else 1

    cells = filter_cells(config, args.cells)
    if not cells:
        print("No hay celdas que procesar.", file=sys.stderr)
        return 1

    print(f"\n=== Preflight campaña — {len(cells)} celdas ===")
    if run_id:
        print(f"Run ID: {run_id} → {artifacts_root(run_id)}")
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
                run_id=run_id,
                improve_profile_override=args.improve_profile,
                hard_negative_csv=hn_csv,
                uniform_ops_per_clip=args.uniform_ops_per_clip,
                fp_category_boost=args.fp_category_boost,
            )
            summary.append(row)
        except Exception as exc:
            print(f"  ERROR {cell['id']}: {exc}", file=sys.stderr)
            summary.append({"cell_id": cell["id"], "error": str(exc)})

    time_rollup: Optional[Dict[str, Any]] = None
    if not args.skip_time_estimate:
        time_rollup = print_campaign_time_rollup(summary, config, run_id=run_id)

    if not getattr(args, "skip_validate", False):
        try:
            from validate_campaign import run_validation, print_report

            print(f"\n{BOLD}=== Validación post-preflight ==={RESET}\n")
            vreport = run_validation(
                config_path=config_path,
                data_root=data_root,
                require_plans=bool(args.write_all),
                run_id=run_id,
                cell_ids=[c["id"] for c in cells],
            )
            print_report(vreport)
            if not vreport.passed:
                return 1
        except ImportError as exc:
            print(f"  {YELLOW}[!] validate_campaign no disponible: {exc}{RESET}")

    if time_rollup is None and not args.skip_time_estimate:
        time_rollup = build_campaign_time_rollup(summary, config)

    master_root = artifacts_root(run_id)
    master_root.mkdir(parents=True, exist_ok=True)
    master = master_root / "preflight_summary.json"
    total_seconds = float((time_rollup or {}).get("total_seconds", 0.0))
    if time_rollup is None:
        total_seconds = sum(
            float(r.get("time_estimate_seconds", 0.0))
            for r in summary
            if "time_estimate_seconds" in r
        )
    payload = {
        "run_id": run_id,
        "config": str(config_path.resolve()) if config_path else str(CONFIG_PATH),
        "experiment_ids": exp_ids,
        "hard_negative_csv": str(hn_csv.resolve()) if hn_csv else None,
        "cells": summary,
        "campaign_time_estimate_seconds": total_seconds,
        "campaign_time_estimate_human": fmt_duration(total_seconds) if total_seconds else None,
        "campaign_time_rollup": time_rollup,
    }
    with open(master, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    if run_id:
        meta = {
            "run_id": run_id,
            "config": payload["config"],
            "hard_negative_csv": payload["hard_negative_csv"],
            "improve_profile_cli": args.improve_profile,
            "uniform_ops_per_clip_cli": args.uniform_ops_per_clip,
            "fp_category_boost_cli": args.fp_category_boost,
        }
        with open(run_meta_path(run_id), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
            f.write("\n")
    print(f"\nResumen: {master}")
    if time_rollup:
        print(
            f"{BOLD}Tiempo total estimado (train, {time_rollup['total_train_runs']} runs): "
            f"{CYAN}{time_rollup['total_human']}{RESET} ({time_rollup['total_hours']} h)"
        )
    elif args.skip_time_estimate:
        print(f"{YELLOW}[!] Estimación omitida (--skip-time-estimate).{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
