#!/usr/bin/env python3
"""
Evaluación campaña mass-augment: métricas val REALES + batería sintética.

Para cada celda:
  1) evaluate_campaign (clips reales en val) → F1 / FP
  2) X clips aleatorios no-robo + X robos, Y variantes sintéticas → FP / recall

Uso:
  python evaluate_mass_augment.py --all --run-id mass_20260707_120000
  python evaluate_mass_augment.py --cells bin_full mc_full --split val
"""
from __future__ import annotations

import argparse
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
    )
    from evaluate_campaign import evaluate_cell
    from evaluate_validation import load_split_uids, build_split_examples
    from preflight_mass_augment import resolve_mass_cells, load_mass_config
    from mass_augment import load_mass_augment_config, run_synthetic_battery_for_model
    from model_config import ROBBERY_CLASS
except ImportError as exc:
    raise SystemExit(f"Import error: {exc}") from exc


def _pick_best_model(reports_dir: Path, split: str) -> Optional[Dict[str, Any]]:
    path = reports_dir / f"{split}_per_model_best.json"
    if not path.is_file():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not rows:
        return None
    return max(rows, key=lambda r: float(r.get("f1_pct") or 0))


def _real_metrics_from_best(best: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not best:
        return {"available": False}
    return {
        "available": True,
        "model": best.get("model"),
        "decision_mode": best.get("decision_mode"),
        "threshold": best.get("threshold"),
        "f1_pct": float(best.get("f1_pct") or 0),
        "recall_pct": float(best.get("recall_pct") or 0),
        "fp_rate_pct": float(best.get("fp_rate_pct") or 0),
        "fn_count": best.get("fn_count"),
        "fp_count": best.get("fp_count"),
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

    real_eval = evaluate_cell(
        cell,
        config,
        split=split,
        export_fp=export_fp,
        run_id=run_id,
    )
    best = _pick_best_model(reports_dir, split)
    real_metrics = _real_metrics_from_best(best)

    synthetic_block: Dict[str, Any] = {"available": False}
    if best and best.get("model"):
        model_path = models_dir / str(best["model"])
        if model_path.is_file():
            with open(plan_path, "r", encoding="utf-8") as f:
                plan = json.load(f)
            mass_block = plan.get("mass_augmentation") or {}
            cfg_path = mass_block.get("config_path")
            syn_cfg = load_mass_augment_config(cfg_path) if cfg_path else dict(mass_cfg)

            split_uids, split_meta = load_split_uids(split_name=split, training_plan_path=plan_path)
            split_meta["split_name"] = split
            val_examples, _pool = build_split_examples(
                split_uids=split_uids,
                split_meta=split_meta,
                pose_source=cell["pose_source"],
                single_user_only=bool(config.get("single_user_only", True)),
                task=cell["task"],
            )

            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            label_to_idx = ckpt["label_to_idx"]
            arch_cfg = dict(ckpt.get("config") or {})
            arch_cfg["seq_len"] = int(ckpt.get("seq_len", 64))

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            syn = run_synthetic_battery_for_model(
                model_path,
                val_examples=val_examples,
                cfg=syn_cfg,
                task=cell["task"],
                label_to_idx=label_to_idx,
                arch_cfg=arch_cfg,
                robbery_class=int(config.get("robbery_class", ROBBERY_CLASS)),
                device=device,
            )
            synthetic_block = {
                "available": True,
                "model": str(best["model"]),
                **syn,
            }

            syn_path = reports_dir / f"{split}_synthetic_eval.json"
            with open(syn_path, "w", encoding="utf-8") as f:
                json.dump(synthetic_block, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(
                f"  [{cell_id}] Sintético — FP no-robo: "
                f"{syn['synthetic_negatives'].get('fp_rate_pct', 0):.2f}% | "
                f"Robo F1: {syn['synthetic_robbery'].get('f1_pct', 0):.1f}%"
            )
        else:
            synthetic_block = {"available": False, "error": f"No existe {model_path}"}
    else:
        synthetic_block = {"available": False, "error": "Sin modelo best en per_model_best"}

    combined = {
        "cell_id": cell_id,
        "task": cell["task"],
        "split": split,
        "real_val": real_metrics,
        "synthetic_val": synthetic_block,
        "evaluate_campaign": real_eval,
    }
    out_path = reports_dir / f"{split}_mass_augment_eval_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(
        f"  [{cell_id}] Real @ {real_metrics.get('model', '?')}: "
        f"F1={real_metrics.get('f1_pct', 0):.1f}% FP={real_metrics.get('fp_rate_pct', 0):.2f}%"
    )
    print(f"  [OK] resumen → {out_path}")
    return combined


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval mass-augment (real + sintético)")
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
    if run_id:
        print(f"Run ID: {run_id}\n")

    summary: List[Dict[str, Any]] = []
    for cell in cells:
        print(f"\n--- {cell['id']} ---")
        try:
            summary.append(
                evaluate_mass_augment_cell(
                    cell,
                    config,
                    mass_cfg,
                    split=args.split,
                    run_id=run_id,
                    export_fp=args.export_fp_videos,
                )
            )
        except Exception as exc:
            print(f"  ERROR {cell['id']}: {exc}", file=sys.stderr)
            summary.append({"cell_id": cell["id"], "error": str(exc)})

    master = master_reports_dir(run_id) / f"{args.split}_mass_augment_eval_cells.json"
    with open(master, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\nResumen master: {master}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
