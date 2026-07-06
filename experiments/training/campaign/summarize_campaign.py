#!/usr/bin/env python3
"""
Consolida CSVs de la campaña en un leaderboard maestro y recomendaciones.

Uso:
  python summarize_campaign.py
"""
from __future__ import annotations

import argparse
import csv
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
    from campaign_paths import load_merged_campaign_config, filter_cells, master_reports_dir, artifacts_root
except ImportError as exc:
    raise SystemExit(f"Import error: {exc}") from exc


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def consolidate(
    split: str = "val",
    run_id: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    config = load_merged_campaign_config(config_path)
    cells = filter_cells(config, None)
    target_rec = float(config.get("target_recall_pct", 60.0))
    target_fp = float(config.get("target_fp_rate_pct", 0.1))
    interactions = int(config.get("interactions_per_day", 1000))

    root = artifacts_root(run_id)
    all_leader: List[Dict[str, Any]] = []
    all_models: List[Dict[str, Any]] = []
    all_ensemble: List[Dict[str, Any]] = []
    best_ensemble_by_cell: List[Dict[str, Any]] = []

    for cell in cells:
        cid = cell["id"]
        reports = root / "reports" / cid
        lb = reports / f"{split}_leaderboard.csv"
        ens = reports / f"{split}_ensemble_grid.csv"
        best_spec = reports / f"{split}_best_ensemble.json"
        for row in _read_csv(lb):
            all_leader.append({**row, "cell_id": cid})
            mode = str(row.get("decision_mode", ""))
            if mode.startswith("ensemble_"):
                all_ensemble.append({**row, "cell_id": cid})
            else:
                all_models.append({**row, "cell_id": cid})
        for row in _read_csv(ens):
            all_ensemble.append({**row, "cell_id": cid})
        if best_spec.is_file():
            spec = json.loads(best_spec.read_text(encoding="utf-8"))
            best_ensemble_by_cell.append({"cell_id": cid, **spec})

    master_dir = master_reports_dir(run_id)
    leader_path = master_dir / f"campaign_leaderboard_{split}.csv"
    _write_csv(leader_path, all_leader)

    models_path = master_dir / f"campaign_models_{split}.csv"
    _write_csv(models_path, all_models)

    ens_path = master_dir / f"campaign_ensemble_{split}.csv"
    _write_csv(ens_path, all_ensemble)

    best_ens_path = master_dir / f"campaign_best_ensemble_{split}.json"
    if best_ensemble_by_cell:
        with open(best_ens_path, "w", encoding="utf-8") as f:
            json.dump(best_ensemble_by_cell, f, indent=2, ensure_ascii=False)
            f.write("\n")

    best_configs: List[Dict[str, Any]] = []
    for row in all_models:
        rec = _float(row, "recall_pct")
        fp = _float(row, "fp_rate_pct")
        f1 = _float(row, "f1_pct")
        if rec >= target_rec and fp <= target_fp:
            best_configs.append({**row, "alarms_per_day": round(fp / 100.0 * interactions, 4)})

    best_configs.sort(key=lambda r: (r.get("fp_rate_pct", 999), -r.get("f1_pct", 0)))
    best_path = master_dir / f"campaign_best_configs_{split}.csv"
    _write_csv(best_path, best_configs)

    gaps_lines: List[str] = []
    if not all_models and not all_ensemble:
        gaps_lines.append("No hay resultados de evaluación. Ejecuta evaluate_campaign.py --all")
    else:
        if all_models:
            top_model = max(all_models, key=lambda r: _float(r, "f1_pct"))
            gaps_lines.append(
                f"Mejor modelo individual ({split}): {top_model.get('cell_id')} / "
                f"{top_model.get('model')} [{top_model.get('decision_mode')}] "
                f"F1={top_model.get('f1_pct')}% FP={top_model.get('fp_rate_pct')}%"
            )
        if best_ensemble_by_cell:
            for spec in best_ensemble_by_cell:
                gaps_lines.append(
                    f"Mejor ensemble {spec.get('cell_id')}: {spec.get('decision_mode')} "
                    f"{spec.get('models')} @ {spec.get('thresholds')} — "
                    f"F1={spec.get('f1_pct')}% Rec={spec.get('recall_pct')}% "
                    f"FP={spec.get('fp_rate_pct')}% (FN={spec.get('fn')} FP={spec.get('fp')})"
                )
        meets = [r for r in all_models if _float(r, "recall_pct") >= target_rec]
        if not meets:
            gaps_lines.append(f"Ningún modelo individual alcanza recall ≥ {target_rec}% en {split}.")
        low_fp = [r for r in meets if _float(r, "fp_rate_pct") <= target_fp]
        if meets and not low_fp:
            best_fp = min(meets, key=lambda r: _float(r, "fp_rate_pct"))
            gaps_lines.append(
                f"Recall OK pero FP alto en modelos. Mejor compromiso: {best_fp.get('cell_id')} / "
                f"{best_fp.get('model')} FP={best_fp.get('fp_rate_pct')}% — revisa ensemble grid."
            )
        if low_fp:
            top = low_fp[0]
            gaps_lines.append(
                f"Candidato operativo (modelo): {top.get('cell_id')} {top.get('model')} "
                f"F1={top.get('f1_pct')}% FP={top.get('fp_rate_pct')}% "
                f"(~{float(top.get('fp_rate_pct', 0)) / 100 * interactions:.2f} alarmas/día)"
            )

    next_steps = [
        "1. Revisar logs/<split>_<celda>_fn_*.txt (robos no detectados) y _fp_*.txt (falsos positivos).",
        "2. Comparar modelos individuales vs mejor ensemble en campaign_best_ensemble_<split>.json.",
        "3. Revisar fp_clips/ por posibles mal etiquetados.",
        "4. Solo usar split test con la config ganadora (no tunear en test).",
    ]

    gaps_path = master_dir / f"campaign_gaps_{split}.txt"
    gaps_path.write_text("\n".join(gaps_lines) + "\n", encoding="utf-8")
    next_path = master_dir / "campaign_next_steps.txt"
    next_path.write_text("\n".join(next_steps) + "\n", encoding="utf-8")

    summary_txt = master_dir / f"campaign_summary_{split}.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"Campaña — split {split}\n")
        f.write(f"Filas modelos: {len(all_models)}\n")
        f.write(f"Filas ensemble grid: {len(all_ensemble)}\n")
        f.write(f"Configs recall≥{target_rec}% y FP≤{target_fp}% (modelos): {len(best_configs)}\n")
        f.write(f"Leaderboard: {leader_path}\n")
        f.write(f"Modelos: {models_path}\n")
        if best_ensemble_by_cell:
            f.write(f"Best ensemble: {best_ens_path}\n")
        f.write(f"Best configs: {best_path}\n")
        for line in gaps_lines:
            f.write(line + "\n")

    print(f"Leaderboard maestro: {leader_path}")
    print(f"Mejores configs: {best_path} ({len(best_configs)} filas)")
    print(f"Gaps: {gaps_path}")
    return {
        "leaderboard": str(leader_path),
        "best_configs": str(best_path),
        "gaps": str(gaps_path),
        "n_best": len(best_configs),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Resumen maestro de campaña")
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--run-id", type=str, default=None, help="Resumir artifacts/runs/<run-id>/")
    args = ap.parse_args()
    consolidate(
        split=args.split,
        run_id=str(args.run_id).strip() if args.run_id else None,
        config_path=Path(args.config) if args.config else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
