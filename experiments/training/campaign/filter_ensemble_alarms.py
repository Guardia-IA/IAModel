#!/usr/bin/env python3
"""
Filtra un CSV de export_ensemble_fp.py → solo alarmas (pred_label=1) con clip + % robo.

Uso:
  cd experiments/training/campaign

  # Imprimir en pantalla (ordenado de mayor a menor %)
  python filter_ensemble_alarms.py artifacts/reports/bin_full_hardened/val_ensemble_fp_mean_modelo_06+modelo_14_t0.68.csv

  # Guardar CSV simple (abrir en Excel/LibreOffice)
  python filter_ensemble_alarms.py RUTA.csv -o alarmas_simple.csv

  # Solo robos reales bien detectados (TP) o solo falsas alarmas (FP)
  python filter_ensemble_alarms.py RUTA.csv --outcome TP
  python filter_ensemble_alarms.py RUTA.csv --outcome FP
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _pct(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    # El export guarda 0.684 (= 68.4%). Si ya viene 68.4, no dividir otra vez.
    return x * 100.0 if x <= 1.0 else x


def _is_alarm(row: Dict[str, str]) -> bool:
    pl = str(row.get("pred_label", "")).strip()
    return pl in ("1", "1.0", "True", "true")


def _clip_label(row: Dict[str, str]) -> str:
    for key in ("clip_video_path", "clip_path", "clip_name", "uid"):
        v = str(row.get(key) or "").strip()
        if v:
            return v
    return "?"


def _model_pct_cols(row: Dict[str, str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in row.items():
        if k.startswith("p_modelo"):
            out[k.replace("p_", "pct_")] = round(_pct(v), 2)
    return out


def filter_rows(
    rows: List[Dict[str, str]],
    *,
    outcome: Optional[str] = None,
    alarms_only: bool = True,
    min_pct: Optional[float] = None,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for row in rows:
        if alarms_only and not _is_alarm(row):
            continue
        oc = str(row.get("outcome", "")).strip().upper()
        if outcome and oc != outcome.upper():
            continue
        pct = round(_pct(row.get("p_mean", row.get("prob_pos", 0))), 2)
        if min_pct is not None and pct < min_pct:
            continue
        simple: Dict[str, Any] = {
            "pct_robo": pct,
            "clip": _clip_label(row),
            "clip_name": row.get("clip_name", ""),
            "outcome": oc,
            "folder_category": row.get("folder_category", ""),
            "uid": row.get("uid", ""),
        }
        simple.update(_model_pct_cols(row))
        selected.append(simple)
    selected.sort(key=lambda r: (-float(r["pct_robo"]), str(r["clip"])))
    return selected


def main() -> int:
    ap = argparse.ArgumentParser(description="Filtra CSV ensemble → clip + % robo")
    ap.add_argument("csv", type=str, help="CSV de export_ensemble_fp.py (--outcomes all recomendado)")
    ap.add_argument("-o", "--output", type=str, default=None, help="CSV de salida simple")
    ap.add_argument(
        "--outcome",
        choices=["TP", "FP", "FN", "TN"],
        default=None,
        help="Filtrar por tipo (TP=robo real detectado, FP=falsa alarma)",
    )
    ap.add_argument(
        "--all-rows",
        action="store_true",
        help="No filtrar por pred_label=1 (incluye todos los clips del CSV)",
    )
    ap.add_argument("--min-pct", type=float, default=None, help="Solo pct_robo >= este valor (ej. 75)")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.is_file():
        print(f"No existe: {path}", file=sys.stderr)
        return 1

    with open(path, "r", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))

    if not raw:
        print("CSV vacío.", file=sys.stderr)
        return 1

    rows = filter_rows(
        raw,
        outcome=args.outcome,
        alarms_only=not args.all_rows,
        min_pct=args.min_pct,
    )

    if not rows:
        hint = (
            "0 filas. ¿Exportaste solo FP (--outcomes fp por defecto)? "
            "Para ver TODAS las alarmas (TP+FP) regenera con:\n"
            "  python export_ensemble_fp.py --split val --outcomes all\n"
            "Luego vuelve a filtrar este script."
        )
        if args.outcome:
            hint = f"0 filas con outcome={args.outcome}. " + hint
        print(hint, file=sys.stderr)
        return 1

    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Escrito: {out.resolve()} ({len(rows)} filas)")
    else:
        print(f"{'% robo':>8}  {'tipo':<4}  clip")
        print("-" * 80)
        for r in rows:
            print(f"{r['pct_robo']:>7.1f}%  {r.get('outcome','?'):<4}  {r['clip']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
