#!/usr/bin/env python3
"""
Añade columna p_verifier (P(confusable)) de la celda bin_verifier_234 a un CSV de ensemble.

Uso:
  python merge_verifier_probs.py \\
      --ensemble-csv artifacts/runs/fp_v1/reports/bin_filtered_hardened/val_ensemble_fp_....csv \\
      --run-id fp_v1 --split val
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

CAMPAIGN_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_DIR))

from campaign_paths import ensure_cell_dirs, filter_cells, load_merged_campaign_config, training_plan_path
from evaluate_validation import build_split_examples, load_split_uids
from evaluate_campaign import collect_binary_predictions
from class_map_utils import apply_class_map_spec, load_class_map
from campaign_paths import class_map_path


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _pick_verifier_model(models_dir: Path) -> Path:
    best = models_dir / "modelo_36.pt"
    if best.is_file():
        return best
    paths = sorted(models_dir.glob("modelo_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No hay modelos en {models_dir}")
    return paths[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble-csv", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--verifier-cell", default="bin_verifier_234")
    ap.add_argument("--model", type=Path, default=None, help="Checkpoint verificador (default: modelo_36 si existe)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    config = load_merged_campaign_config(path=CAMPAIGN_DIR / "campaign_config_fp_pipeline.json")
    cells = filter_cells(config, [args.verifier_cell])
    if not cells:
        raise SystemExit(f"Celda verificador no encontrada: {args.verifier_cell}")
    cell = cells[0]

    v_arts = ensure_cell_dirs(args.verifier_cell, run_id=args.run_id)
    model_path = args.model or _pick_verifier_model(v_arts["models_dir"])
    plan_path = training_plan_path(args.verifier_cell, run_id=args.run_id)

    split_uids, split_meta = load_split_uids(split_name=args.split, training_plan_path=plan_path)
    split_meta["split_name"] = args.split
    examples, _ = build_split_examples(
        split_uids=split_uids,
        split_meta=split_meta,
        pose_source=cell["pose_source"],
        single_user_only=bool(config.get("single_user_only", True)),
        task=cell["task"],
    )
    cmap = load_class_map(class_map_path(cell["class_map_id"]))
    examples = apply_class_map_spec(examples, cmap)

    records, _yt, probs, _margins = collect_binary_predictions(
        model_path,
        examples,
        training_plan_path=plan_path,
        split_name=args.split,
    )
    # En verificador: positivo=confusable (cats 2/3/14 mapeadas a binario 0, robo=1)
    # Queremos P(confusable): en binario verificador, clase 0 = no-robo del verificador = confusable
    # Wait - binary task: robbery_class=6 is positive (1), confusables are negative (0)
    # So prob_pos = P(robbery) in verifier = NOT what we want
    # We want P(confusable) = P(class 0) = 1 - prob_pos IF verifier is trained robo=1 vs confusable=0
    # Actually verifier: positive=robbery (6), negative=confusable (2,3,14)
    # P(confusable) = 1 - prob_pos (probability of NOT robbery in verifier space = looks like confusable)

    uid_to_p_conf: Dict[str, float] = {}
    for rec, p in zip(records, probs):
        uid_to_p_conf[str(rec["uid"])] = float(1.0 - p)

    rows_in = _read_csv(args.ensemble_csv)
    out_rows: List[Dict[str, Any]] = []
    for row in rows_in:
        uid = row.get("uid") or row.get("pose_path") or ""
        p_conf = uid_to_p_conf.get(uid, uid_to_p_conf.get(row.get("uid_absolute", ""), 0.0))
        out_rows.append({**row, "p_verifier": round(p_conf, 6)})

    out_path = args.out or args.ensemble_csv.with_name(
        args.ensemble_csv.stem + "_with_verifier.csv"
    )
    fields = list(out_rows[0].keys()) if out_rows else ["uid", "p_verifier"]
    _write_csv(out_path, out_rows, fields)
    print(json.dumps({"out": str(out_path), "verifier_model": str(model_path), "rows": len(out_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
