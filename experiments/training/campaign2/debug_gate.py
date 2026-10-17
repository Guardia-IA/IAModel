#!/usr/bin/env python3
"""Comprueba que un RUN debug/matrix cumple criterios mínimos."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CAMPAIGN2_DIR = Path(__file__).resolve().parent
for p in (CAMPAIGN2_DIR.parent, CAMPAIGN2_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from campaign2_paths import read_current_run_id, run_layout  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--min-robbery-splits", type=int, default=1)
    args = ap.parse_args()
    run_id = args.run_id or read_current_run_id()
    if not run_id:
        print("ERROR: sin RUN_ID", file=sys.stderr)
        return 2

    layout = run_layout(run_id)
    errors: list[str] = []
    warnings: list[str] = []

    from manifest_paths import discover_manifest_by_pose, load_run_meta

    meta = load_run_meta(run_id)
    man_paths: list[Path] = []
    for _ps, pstr in (meta.get("manifest_by_pose") or {}).items():
        man_paths.append(Path(str(pstr)))
    if not man_paths:
        man_paths = list(discover_manifest_by_pose(layout["cache_dir"]).values())
    if not man_paths:
        man_paths = [layout["cache_dir"] / "manifest.json"]
    robbery_in_eval = 0
    cat6 = 0
    for mp in man_paths:
        if not mp.is_file():
            continue
        man = json.loads(mp.read_text(encoding="utf-8"))
        for e in man.get("entries") or []:
            if int(e.get("label", -1)) == 6:
                cat6 += 1
            if e.get("is_robbery") and str(e.get("split")) in (
                "val_dev",
                "threshold_calib",
            ):
                robbery_in_eval += 1
    if cat6 < 1:
        errors.append("Sin clips cat.6 (robo) en manifest")
    if robbery_in_eval < args.min_robbery_splits:
        errors.append(
            f"Pocos robos en val_dev/threshold_calib ({robbery_in_eval} "
            f"< {args.min_robbery_splits})"
        )

    train_ok = list(layout["jobs_dir"].glob("*/p1_train_result.json"))
    if not train_ok:
        errors.append("Sin p1_train_result.json (train no completado)")
    else:
        for rpath in train_ok:
            ckpt = rpath.parent / "best.pt"
            if not ckpt.is_file():
                errors.append(f"Falta best.pt en {rpath.parent.name}")

    if not (layout["reports_dir"] / "hard_neg_manifest.json").is_file():
        warnings.append("Falta reports/hard_neg_manifest.json (¿preflight sin inventario?)")

    eval_preds = list(layout["jobs_dir"].glob("eval_*/clip_predictions.json"))
    if not eval_preds:
        warnings.append("Sin eval_*/clip_predictions.json (opcional si aún no p1-eval)")

    print(f"DEBUG GATE — {run_id}")
    for w in warnings:
        print(f"  WARN: {w}")
    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        return 1
    print("  OK: criterios mínimos cumplidos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
