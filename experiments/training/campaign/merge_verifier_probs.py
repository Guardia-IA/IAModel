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
    paths = sorted(models_dir.glob("modelo_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No hay modelos en {models_dir}")
    return paths[0]


def _normalize_model_filename(name: str) -> str:
    name = str(name).strip()
    if name.isdigit():
        return f"modelo_{int(name):02d}.pt"
    if not name.endswith(".pt"):
        return f"{name}.pt" if name.startswith("modelo_") else f"modelo_{name}.pt"
    return name


def resolve_verifier_model_path(
    *,
    run_id: str,
    verifier_cell: str,
    split: str,
    explicit: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"No existe checkpoint verificador: {explicit}")
        return explicit.resolve()

    v_arts = ensure_cell_dirs(verifier_cell, run_id=run_id)
    models_dir = v_arts["models_dir"]
    reports_dir = v_arts["reports_dir"]

    from evaluate_campaign import load_best_ensemble_spec

    spec = load_best_ensemble_spec(reports_dir, split)
    if spec and spec.get("models"):
        for raw in spec["models"]:
            candidate = models_dir / _normalize_model_filename(str(raw))
            if candidate.is_file():
                return candidate.resolve()

    per_model_path = reports_dir / f"{split}_per_model_best.json"
    if per_model_path.is_file():
        with open(per_model_path, encoding="utf-8") as f:
            per_model = json.load(f)
        if per_model:
            best = max(per_model, key=lambda row: float(row.get("f1_pct") or 0))
            label = best.get("model") or best.get("label")
            if label:
                candidate = models_dir / _normalize_model_filename(str(label))
                if candidate.is_file():
                    return candidate.resolve()

    return _pick_verifier_model(models_dir).resolve()


def merge_verifier_csv(
    ensemble_csv: Path,
    *,
    run_id: str,
    split: str = "val",
    verifier_cell: str = "bin_verifier_234",
    model: Optional[Path] = None,
    out: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    cfg_path = config_path or (CAMPAIGN_DIR / "campaign_config_fp_pipeline.json")
    config = load_merged_campaign_config(path=cfg_path)
    cells = filter_cells(config, [verifier_cell])
    if not cells:
        raise ValueError(f"Celda verificador no encontrada: {verifier_cell}")
    cell = cells[0]

    model_path = resolve_verifier_model_path(
        run_id=run_id,
        verifier_cell=verifier_cell,
        split=split,
        explicit=model,
        config_path=cfg_path,
    )
    plan_path = training_plan_path(verifier_cell, run_id=run_id)

    split_uids, split_meta = load_split_uids(split_name=split, training_plan_path=plan_path)
    split_meta["split_name"] = split
    examples, _ = build_split_examples(
        split_uids=split_uids,
        split_meta=split_meta,
        pose_source=cell["pose_source"],
        single_user_only=bool(config.get("single_user_only", True)),
        task=cell["task"],
        positive_class=int(config.get("robbery_class", 6)),
    )
    # No aplicar class_map aquí: build_split_examples(task=binary) ya deja labels 0/1.
    # apply_class_map_spec relabelaría confusables (2/3/14) → KeyError en label_to_idx del checkpoint.

    records, _yt, probs, _margins = collect_binary_predictions(
        model_path,
        examples,
        training_plan_path=plan_path,
        split_name=split,
    )
    # Verificador binario: positivo=robo (6), negativo=confusable (2/3/14) → P(confusable)=1-P(robo)
    uid_to_p_conf: Dict[str, float] = {}
    for rec, p in zip(records, probs):
        uid_to_p_conf[str(rec["uid"])] = float(1.0 - p)

    rows_in = _read_csv(ensemble_csv)
    out_rows: List[Dict[str, Any]] = []
    for row in rows_in:
        uid = row.get("uid") or row.get("pose_path") or ""
        p_conf = uid_to_p_conf.get(uid, uid_to_p_conf.get(row.get("uid_absolute", ""), 0.0))
        out_rows.append({**row, "p_verifier": round(p_conf, 6)})

    out_path = out or ensemble_csv.with_name(ensemble_csv.stem + "_with_verifier.csv")
    fields = list(out_rows[0].keys()) if out_rows else ["uid", "p_verifier"]
    _write_csv(out_path, out_rows, fields)
    return {
        "out": out_path,
        "verifier_model": model_path,
        "rows": len(out_rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble-csv", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--verifier-cell", default="bin_verifier_234")
    ap.add_argument("--model", type=Path, default=None, help="Checkpoint verificador (default: mejor del eval)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    result = merge_verifier_csv(
        args.ensemble_csv,
        run_id=args.run_id,
        split=args.split,
        verifier_cell=args.verifier_cell,
        model=args.model,
        out=args.out,
    )
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in result.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
