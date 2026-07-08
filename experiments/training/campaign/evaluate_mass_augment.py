#!/usr/bin/env python3
"""
Evaluación mass-augment:
  A) Val REAL (solo clips reales del split val/test)
  B) Batería sintética (X clips × Y ops) sobre pool val real
  C) CSVs: best F1, best min FP, best ensemble (real + sintético)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

CAMPAIGN_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_DIR))

try:
    from campaign_paths import (
        load_merged_campaign_config,
        ensure_cell_dirs,
        training_plan_path,
        master_reports_dir,
        CONFIG_PATH,
        resolve_mass_cells,
    )
    from evaluate_campaign import evaluate_cell
    from evaluate_validation import load_split_uids, build_split_examples
    from preflight_mass_augment import load_mass_config
    from mass_augment import (
        load_mass_augment_config,
        run_synthetic_battery_for_model,
        run_synthetic_battery_for_ensemble,
    )
    from mass_augment_support import (
        verify_mass_augment_split,
        selection_specs_from_eval,
        write_csv,
        fp_by_category_from_manifest,
        synthetic_gate_check,
    )
    from calibration_utils import calibrate_model_on_val
    from model_config import ROBBERY_CLASS
except ImportError as exc:
    raise SystemExit(f"Import error: {exc}") from exc


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _real_row_from_spec(cell_id: str, split: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cell_id": cell_id,
        "split": split,
        "eval_pool": "real_val_clips",
        "selection": spec.get("selection"),
        "kind": spec.get("kind"),
        "model": spec.get("model") or "|".join(spec.get("models") or []),
        "threshold": spec.get("threshold") or spec.get("thresholds"),
        "f1_pct": spec.get("f1_pct"),
        "recall_pct": spec.get("recall_pct"),
        "fp_rate_pct": spec.get("fp_rate_pct"),
    }


def _synthetic_rows_from_battery(
    cell_id: str,
    split: str,
    spec: Dict[str, Any],
    battery: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base = {
        "cell_id": cell_id,
        "split": split,
        "eval_pool": "synthetic_augmented",
        "selection": spec.get("selection"),
        "kind": spec.get("kind"),
        "model": spec.get("model") or "|".join(spec.get("models") or []),
        "clips_x": battery.get("clips_x"),
        "variants_y": battery.get("variants_y"),
    }
    neg = battery.get("synthetic_negatives") or {}
    pos = battery.get("synthetic_robbery") or {}
    rows.append({
        **base,
        "battery_part": "non_robbery_fp_test",
        "n": neg.get("n"),
        "fp": neg.get("fp"),
        "fp_rate_pct": neg.get("fp_rate_pct"),
        "f1_pct": "",
        "recall_pct": "",
    })
    rows.append({
        **base,
        "battery_part": "robbery_detection",
        "n": pos.get("n"),
        "fp": pos.get("fp"),
        "fp_rate_pct": pos.get("fp_rate_pct"),
        "f1_pct": pos.get("f1_pct"),
        "recall_pct": pos.get("recall_pct"),
    })
    return rows


def _verify_val_examples_real(examples: List[Any]) -> Dict[str, Any]:
    forced = sum(1 for ex in examples if getattr(ex, "forced_ops", None) is not None)
    return {
        "n_examples": len(examples),
        "n_with_forced_ops": forced,
        "all_real_clips": forced == 0,
    }


def evaluate_mass_augment_cell(
    cell: Dict[str, Any],
    config: Dict[str, Any],
    mass_cfg: Dict[str, Any],
    *,
    split: str = "val",
    run_id: Optional[str] = None,
    export_fp: bool = False,
) -> Dict[str, Any]:
    cell_id = cell["id"]
    arts = ensure_cell_dirs(cell_id, run_id=run_id)
    reports_dir = arts["reports_dir"]
    models_dir = arts["models_dir"]
    plan_path = training_plan_path(cell_id, run_id=run_id)

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    split_check = verify_mass_augment_split(plan)
    if not split_check.get("passed_disjoint"):
        raise ValueError(f"Split inválido en {cell_id}: {split_check.get('issues')}")

    real_eval = evaluate_cell(
        cell, config, split=split, export_fp=export_fp, run_id=run_id,
    )

    split_uids, split_meta = load_split_uids(split_name=split, training_plan_path=plan_path)
    split_meta["split_name"] = split
    split_meta["split_uids_all"] = {
        k: [str(x) for x in v] for k, v in plan.get("split_uids", {}).items()
    }
    val_examples, pool_info = build_split_examples(
        split_uids=split_uids,
        split_meta=split_meta,
        pose_source=cell["pose_source"],
        single_user_only=bool(config.get("single_user_only", True)),
        task=cell["task"],
    )
    val_real_check = _verify_val_examples_real(val_examples)

    mass_block = plan.get("mass_augmentation") or {}
    cfg_path = mass_block.get("config_path")
    syn_cfg = load_mass_augment_config(cfg_path) if cfg_path else dict(mass_cfg)

    specs = selection_specs_from_eval(reports_dir, split, config, models_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    apply_cal = bool((config.get("mass_augment") or {}).get("apply_temperature_calibration", True))
    gate_ratio = float((config.get("mass_augment") or {}).get("synthetic_gate_max_ratio", 2.0))

    fp_manifest = reports_dir / f"{split}_fp_manifest.csv"
    fp_by_cat = fp_by_category_from_manifest(fp_manifest)
    fp_cat_path = reports_dir / f"{split}_fp_by_category.csv"
    write_csv(fp_cat_path, fp_by_cat)

    real_csv_rows: List[Dict[str, Any]] = []
    synthetic_csv_rows: List[Dict[str, Any]] = []
    calibrated_rows: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []
    synthetic_by_selection: Dict[str, Any] = {}

    for spec in specs:
        work = dict(spec)
        real_csv_rows.append(_real_row_from_spec(cell_id, split, work))

        thr = float(work.get("threshold") or 0.5)
        temp = 1.0
        if apply_cal and work.get("kind") == "single_model":
            mp_cal = Path(work["model_path"])
            if mp_cal.is_file():
                ck_cal = torch.load(mp_cal, map_location="cpu", weights_only=False)
                cal = calibrate_model_on_val(
                    mp_cal,
                    val_examples,
                    label_to_idx=ck_cal["label_to_idx"],
                    task=cell["task"],
                    robbery_class=int(config.get("robbery_class", ROBBERY_CLASS)),
                    device=device,
                )
                temp = float(cal.get("temperature", 1.0))
                thr = float(cal.get("calibrated_threshold", thr))
                work["temperature"] = temp
                work["calibrated_threshold"] = thr
                work["calibrated_f1_pct"] = cal.get("f1_pct")
                work["calibrated_fp_rate_pct"] = cal.get("fp_rate_pct")
                work["calibrated_recall_pct"] = cal.get("recall_pct")
                calibrated_rows.append({
                    "cell_id": cell_id,
                    "split": split,
                    "selection": work.get("selection"),
                    "model": work.get("model"),
                    "temperature": temp,
                    "threshold": thr,
                    "f1_pct": cal.get("f1_pct"),
                    "fp_rate_pct": cal.get("fp_rate_pct"),
                    "recall_pct": cal.get("recall_pct"),
                })

        if work.get("kind") == "single_model":
            mp = Path(work["model_path"])
            if not mp.is_file():
                synthetic_by_selection[work["selection"]] = {"error": f"No existe {mp}"}
                continue
            ckpt = torch.load(mp, map_location="cpu", weights_only=False)
            arch_cfg = dict(ckpt.get("config") or {})
            arch_cfg["seq_len"] = int(ckpt.get("seq_len", 64))
            battery = run_synthetic_battery_for_model(
                mp,
                val_examples=val_examples,
                cfg=syn_cfg,
                task=cell["task"],
                label_to_idx=ckpt["label_to_idx"],
                arch_cfg=arch_cfg,
                robbery_class=int(config.get("robbery_class", ROBBERY_CLASS)),
                device=device,
                threshold=thr,
                temperature=temp,
            )
        elif work.get("kind") == "ensemble":
            paths = [Path(p) for p in work.get("model_paths") or []]
            paths = [p for p in paths if p.is_file()]
            if not paths:
                synthetic_by_selection[work["selection"]] = {"error": "Modelos ensemble no encontrados"}
                continue
            battery = run_synthetic_battery_for_ensemble(
                paths,
                val_examples=val_examples,
                cfg=syn_cfg,
                task=cell["task"],
                rule=str(work.get("rule") or "mean"),
                thresholds=work.get("thresholds"),
                robbery_class=int(config.get("robbery_class", ROBBERY_CLASS)),
                device=device,
            )
        else:
            continue

        synthetic_by_selection[work["selection"]] = battery
        synthetic_csv_rows.extend(_synthetic_rows_from_battery(cell_id, split, work, battery))

        neg = battery.get("synthetic_negatives") or {}
        gate = synthetic_gate_check(
            float(work.get("calibrated_fp_rate_pct") or work.get("fp_rate_pct") or 0),
            float(neg.get("fp_rate_pct") or 0),
            max_ratio=gate_ratio,
        )
        gate_rows.append({
            "cell_id": cell_id,
            "selection": work.get("selection"),
            "model": work.get("model") or "|".join(work.get("models") or []),
            **gate,
        })

    real_csv_path = reports_dir / f"{split}_mass_augment_real_picks.csv"
    syn_csv_path = reports_dir / f"{split}_mass_augment_synthetic_picks.csv"
    cal_csv_path = reports_dir / f"{split}_mass_augment_calibrated.csv"
    gate_csv_path = reports_dir / f"{split}_mass_augment_synthetic_gate.csv"
    write_csv(real_csv_path, real_csv_rows)
    write_csv(syn_csv_path, synthetic_csv_rows)
    write_csv(cal_csv_path, calibrated_rows)
    write_csv(gate_csv_path, gate_rows)

    combined = {
        "cell_id": cell_id,
        "task": cell["task"],
        "split": split,
        "split_verification": split_check,
        "val_real_clips_check": val_real_check,
        "pool_info": pool_info,
        "selections": specs,
        "fp_by_category_csv": str(fp_cat_path),
        "real_picks_csv": str(real_csv_path),
        "synthetic_picks_csv": str(syn_csv_path),
        "calibrated_csv": str(cal_csv_path),
        "synthetic_gate_csv": str(gate_csv_path),
        "synthetic_by_selection": synthetic_by_selection,
        "synthetic_gates": gate_rows,
        "evaluate_campaign": real_eval,
    }
    out_path = reports_dir / f"{split}_mass_augment_eval_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"  [{cell_id}] Split OK: train={split_check['uids_train']} val={split_check['uids_val']} test={split_check['uids_test']}")
    print(f"  [{cell_id}] Val real clips: {val_real_check['n_examples']} (forced_ops={val_real_check['n_with_forced_ops']})")
    for spec in specs:
        sel = spec.get("selection")
        print(
            f"  [{cell_id}] Real {sel}: F1={spec.get('f1_pct', 0):.1f}% "
            f"FP={spec.get('fp_rate_pct', 0):.3f}% Rec={spec.get('recall_pct', 0):.1f}%"
        )
        bat = synthetic_by_selection.get(sel) or {}
        neg = bat.get("synthetic_negatives") or {}
        pos = bat.get("synthetic_robbery") or {}
        if neg or pos:
            print(
                f"           Sintético {sel}: FP_no_robo={neg.get('fp_rate_pct', '?')}% | "
                f"Robo F1={pos.get('f1_pct', '?')}% Rec={pos.get('recall_pct', '?')}%"
            )
    for g in gate_rows:
        if g.get("warning"):
            print(f"  [{cell_id}] GATE {g.get('selection')}: {g['warning']}")
    print(f"  [OK] CSV real → {real_csv_path}")
    print(f"  [OK] CSV sintético → {syn_csv_path}")
    print(f"  [OK] FP/categoría → {fp_cat_path}")
    return combined


def consolidate_master_csvs(
    summary: List[Dict[str, Any]],
    *,
    split: str,
    run_id: Optional[str],
) -> Dict[str, str]:
    master = master_reports_dir(run_id)
    all_real: List[Dict[str, Any]] = []
    all_syn: List[Dict[str, Any]] = []
    all_cal: List[Dict[str, Any]] = []
    all_gates: List[Dict[str, Any]] = []
    all_fp_cat: List[Dict[str, Any]] = []
    comparison: List[Dict[str, Any]] = []
    deploy_rows: List[Dict[str, Any]] = []

    for item in summary:
        if item.get("error"):
            continue
        cid = item["cell_id"]
        for p, dest in (
            (item.get("real_picks_csv"), all_real),
            (item.get("synthetic_picks_csv"), all_syn),
            (item.get("calibrated_csv"), all_cal),
            (item.get("synthetic_gate_csv"), all_gates),
            (item.get("fp_by_category_csv"), all_fp_cat),
        ):
            if p:
                for row in _read_csv(Path(str(p))):
                    dest.append({**row, "cell_id": cid})

        for spec in item.get("selections") or []:
            sel = spec.get("selection")
            bat = (item.get("synthetic_by_selection") or {}).get(sel) or {}
            neg = bat.get("synthetic_negatives") or {}
            pos = bat.get("synthetic_robbery") or {}
            comparison.append({
                "cell_id": cid,
                "split": split,
                "selection": sel,
                "model": spec.get("model") or "|".join(spec.get("models") or []),
                "real_f1_pct": spec.get("f1_pct"),
                "real_fp_rate_pct": spec.get("fp_rate_pct"),
                "real_recall_pct": spec.get("recall_pct"),
                "synthetic_fp_rate_pct": neg.get("fp_rate_pct"),
                "synthetic_robbery_f1_pct": pos.get("f1_pct"),
                "synthetic_robbery_recall_pct": pos.get("recall_pct"),
                "synthetic_clips_x": bat.get("clips_x"),
                "synthetic_variants_y": bat.get("variants_y"),
            })
            if sel in ("best_operational", "best_min_fp", "best_ensemble"):
                deploy_rows.append({
                    "cell_id": cid,
                    "selection": sel,
                    "model": spec.get("model") or "|".join(spec.get("models") or []),
                    "f1_pct": spec.get("f1_pct"),
                    "fp_rate_pct": spec.get("fp_rate_pct"),
                    "recall_pct": spec.get("recall_pct"),
                })

    paths = {
        "real_picks": master / f"{split}_mass_augment_real_picks_all_cells.csv",
        "synthetic_picks": master / f"{split}_mass_augment_synthetic_picks_all_cells.csv",
        "calibrated": master / f"{split}_mass_augment_calibrated_all_cells.csv",
        "synthetic_gates": master / f"{split}_mass_augment_synthetic_gate_all_cells.csv",
        "fp_by_category": master / f"{split}_fp_by_category_all_cells.csv",
        "comparison": master / f"{split}_mass_augment_real_vs_synthetic.csv",
        "deploy_candidates": master / f"{split}_mass_augment_deploy_candidates.csv",
    }
    write_csv(paths["real_picks"], all_real)
    write_csv(paths["synthetic_picks"], all_syn)
    write_csv(paths["calibrated"], all_cal)
    write_csv(paths["synthetic_gates"], all_gates)
    write_csv(paths["fp_by_category"], all_fp_cat)
    write_csv(paths["comparison"], comparison)
    write_csv(paths["deploy_candidates"], deploy_rows)
    return {k: str(v) for k, v in paths.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval mass-augment (real + sintético + CSVs)")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--split", type=str, default="val", choices=["val", "test"])
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--export-fp-videos", action="store_true")
    args = ap.parse_args()

    config = load_merged_campaign_config(Path(args.config) if args.config else None)
    if args.all:
        cells = resolve_mass_cells(config, None)
    elif args.cells:
        cells = resolve_mass_cells(config, args.cells)
    else:
        print("Indica --all o --cells.", file=sys.stderr)
        return 1

    run_id = str(args.run_id).strip() if args.run_id else None
    mass_cfg = load_mass_config(config)

    print(f"\n=== Eval MASS AUG — {len(cells)} celdas, split={args.split} ===")
    print("  A) Métricas en clips REALES del split (sin augment en val)")
    print("  B) Batería sintética: X clips val × Y ops augmentadas\n")

    summary: List[Dict[str, Any]] = []
    for cell in cells:
        print(f"\n--- {cell['id']} ---")
        try:
            summary.append(
                evaluate_mass_augment_cell(
                    cell, config, mass_cfg,
                    split=args.split, run_id=run_id, export_fp=args.export_fp_videos,
                )
            )
        except Exception as exc:
            print(f"  ERROR {cell['id']}: {exc}", file=sys.stderr)
            summary.append({"cell_id": cell["id"], "error": str(exc)})

    csv_paths = consolidate_master_csvs(summary, split=args.split, run_id=run_id)
    master = master_reports_dir(run_id) / f"{args.split}_mass_augment_eval_cells.json"
    with open(master, "w", encoding="utf-8") as f:
        json.dump({"cells": summary, "master_csvs": csv_paths}, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\n=== CSVs maestros ===")
    for k, p in csv_paths.items():
        print(f"  {k}: {p}")
    print(f"  JSON: {master}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
