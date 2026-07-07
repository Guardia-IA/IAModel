#!/usr/bin/env python3
"""Resumen legible post-eval mass augment + recomendaciones de despliegue."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

CAMPAIGN_DIR = Path(__file__).resolve().parent
if str(CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_DIR))

from campaign_paths import master_reports_dir, load_merged_campaign_config, CONFIG_PATH


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run_id = str(args.run_id).strip() if args.run_id else None
    config = load_merged_campaign_config(Path(args.config) if args.config else CONFIG_PATH)
    ma = config.get("mass_augment") or {}
    deploy = ma.get("deploy_targets") or {}
    target_f1 = float(deploy.get("f1_pct", 70.0))
    target_fp = float(deploy.get("fp_rate_pct", 0.01))
    target_rec = float(deploy.get("recall_pct", 60.0))

    master = master_reports_dir(run_id)
    split = args.split
    cmp_path = master / f"{split}_mass_augment_real_vs_synthetic.csv"
    deploy_path = master / f"{split}_mass_augment_deploy_candidates.csv"
    gates_path = master / f"{split}_mass_augment_synthetic_gate_all_cells.csv"
    fp_cat_path = master / f"{split}_fp_by_category_all_cells.csv"

    lines = [
        f"Mass augment — resumen eval split={split}",
        f"Objetivos despliegue: F1≥{target_f1}% | recall≥{target_rec}% | FP≤{target_fp}%",
        "",
        "IMPORTANTE: tunear solo en VAL. Usa TEST una sola vez al final con la config ganadora.",
        "",
        f"Comparación real vs sintético: {cmp_path}",
        f"Candidatos despliegue: {deploy_path}",
        f"Gates sintéticos: {gates_path}",
        f"FP por categoría: {fp_cat_path}",
        "",
        "=== Mejores por celda (real val) ===",
    ]

    comparison = _read_csv(cmp_path)
    for row in comparison:
        if row.get("selection") not in ("best_f1", "best_min_fp", "best_operational", "best_ensemble"):
            continue
        lines.append(
            f"  {row.get('cell_id')} [{row.get('selection')}] "
            f"real F1={row.get('real_f1_pct')}% FP={row.get('real_fp_rate_pct')}% Rec={row.get('real_recall_pct')}% | "
            f"syn FP={row.get('synthetic_fp_rate_pct')}% roboF1={row.get('synthetic_robbery_f1_pct')}%"
        )

    lines.extend(["", "=== Synthetic gates (overfit augment) ==="])
    for g in _read_csv(gates_path):
        if g.get("warning"):
            lines.append(f"  WARN {g.get('cell_id')} {g.get('selection')}: {g.get('warning')}")
        elif g.get("passed") == "True" or g.get("passed") is True:
            lines.append(f"  OK   {g.get('cell_id')} {g.get('selection')}: ratio={g.get('ratio')}")

    lines.extend(["", "=== Top FP por categoría (val) ==="])
    fp_rows = sorted(_read_csv(fp_cat_path), key=lambda r: -_f(r, "fp_count"))[:12]
    for r in fp_rows:
        lines.append(
            f"  cat {r.get('folder_category')}: {r.get('fp_count')} FP ({r.get('fp_pct_of_total')}%)"
        )

    lines.extend(["", "=== Candidatos operativos (priorizar best_operational / hardened) ==="])
    deploy = _read_csv(deploy_path)
    deploy.sort(key=lambda r: (_f(r, "fp_rate_pct"), -_f(r, "f1_pct")))
    for r in deploy[:12]:
        ok = _f(r, "f1_pct") >= target_f1 and _f(r, "fp_rate_pct") <= target_fp
        tag = "OK" if ok else "  "
        lines.append(
            f"  {tag} {r.get('cell_id')} [{r.get('selection')}] {r.get('model')} "
            f"F1={r.get('f1_pct')}% FP={r.get('fp_rate_pct')}% Rec={r.get('recall_pct')}%"
        )

    lines.extend([
        "",
        "Comparar con campaña de ayer (datos reales): columna real_* en comparison CSV.",
        "Si mass_aug mejora real FP pero empeora synthetic gate → revisar augment cat. 2/14.",
    ])

    out = master / f"{split}_mass_augment_summary.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nGuardado: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
