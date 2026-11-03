#!/usr/bin/env python3
"""
Ensayo corto del mismo camino que la campaña: N clips, pocas épocas, train + val + test.

Escribe en artifacts/runs/<RUN_ID>_debug/ y no toca los modelos del run de producción.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

CAMPAIGN_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CAMPAIGN_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_DIR))

from campaign_paths import (  # noqa: E402
    category_aug_path,
    ensure_cell_dirs,
    filter_cells,
    load_merged_campaign_config,
    resolve_experiment_ids,
    training_plan_path,
)
from evaluate_campaign import evaluate_cell  # noqa: E402
from model_config import (  # noqa: E402
    MAX_OCCLUSION_RATIO,
    MIN_CLIP_SECONDS,
    MIN_VALID_FRAMES,
    MIN_VALID_PCT,
    SEED,
)
from preflight_train_plan import write_training_plan  # noqa: E402
from train_campaign import train_cell  # noqa: E402
from train_model_operations import (  # noqa: E402
    _example_folder_category,
    collect_debug_examples,
    count_examples_by_folder_category,
    split_uids_from_example_lists,
)


def _split_debug(examples: List[Any]) -> tuple:
    n = len(examples)
    if n < 3:
        raise SystemExit(
            f"Solo hay {n} clips válidos. Hacen falta al menos 3 para train, val y test."
        )
    n_test = max(1, int(round(n * 0.2)))
    n_val = max(1, int(round(n * 0.2)))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        remain = n - 1
        n_val = max(1, remain // 2)
        n_test = remain - n_val
    by_cat: Dict[int, List[Any]] = {}
    for ex in examples:
        by_cat.setdefault(int(_example_folder_category(ex)), []).append(ex)
    cats = sorted(by_cat)
    flat: List[Any] = []
    while any(by_cat[c] for c in cats):
        for cat in cats:
            if by_cat[cat]:
                flat.append(by_cat[cat].pop(0))
    train = flat[:n_train]
    val = flat[n_train : n_train + n_val]
    test = flat[n_train + n_val :]
    return train, val, test


def _counts(examples: List[Any]) -> Dict[str, int]:
    return {str(k): int(v) for k, v in count_examples_by_folder_category(examples).items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Ensayo de campaña con N clips")
    ap.add_argument("--n", type=int, default=10, help="Clips en total (train+val+test)")
    ap.add_argument("--source-run-id", type=str, default=None, help="Run del preflight")
    ap.add_argument("--run-id", type=str, default=None, help="Se interpreta como el run del preflight")
    ap.add_argument("--cell", type=str, default="bin_full")
    ap.add_argument("--exp-id", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument(
        "--data-augmentation",
        action="store_true",
        help="No cambia este ensayo: el augment es el del plan del preflight.",
    )
    ap.add_argument("--data-root", type=str, default=None)
    args = ap.parse_args()

    source = (args.source_run_id or args.run_id or "").strip()
    if not source:
        print("Indica el RUN_ID del preflight (--source-run-id o --run-id).", file=sys.stderr)
        return 1
    if args.n < 3:
        print("--n tiene que ser al menos 3 (train, val y test).", file=sys.stderr)
        return 1

    dest = f"{source}_debug"
    config = load_merged_campaign_config(Path(args.config) if args.config else None)
    cells = filter_cells(config, [args.cell])
    if not cells:
        cells = filter_cells(config, None)
        cell = next((c for c in cells if c["id"] == "bin_full"), cells[0] if cells else None)
    else:
        cell = cells[0]
    if cell is None:
        print("No hay celdas en la config.", file=sys.stderr)
        return 1

    src_plan_path = training_plan_path(cell["id"], run_id=source)
    if not src_plan_path.is_file():
        print(
            f"No está el plan de {cell['id']} en {src_plan_path}. "
            "Lanza antes el preflight de ese RUN_ID.",
            file=sys.stderr,
        )
        return 1
    plan = json.loads(src_plan_path.read_text(encoding="utf-8"))
    filters = plan.get("filters") or {}
    skip = {int(x) for x in (plan.get("class_map") or {}).get("exclude") or []}
    data_root = Path(args.data_root) if args.data_root else None
    if data_root is None and plan.get("data_root"):
        data_root = Path(str(plan["data_root"]))

    print(
        f"[DEBUG] Run de producción {source} no se toca. Artefactos en {dest}.",
        flush=True,
    )
    print(
        f"[DEBUG] Celda {cell['id']} | {args.n} clips | {args.epochs} épocas | "
        "val y test con los clips de este ensayo.",
        flush=True,
    )
    examples = collect_debug_examples(
        n=int(args.n),
        pose_source=str(plan.get("pose_source") or cell["pose_source"]),
        single_user_only=bool(plan.get("single_user_only", True)),
        min_clip_seconds=float(filters.get("min_clip_seconds", MIN_CLIP_SECONDS)),
        min_valid_frames=int(filters.get("min_valid_frames", MIN_VALID_FRAMES)),
        min_valid_pct=float(filters.get("min_valid_pct", MIN_VALID_PCT)),
        max_occlusion_ratio=float(filters.get("max_occlusion_ratio", MAX_OCCLUSION_RATIO)),
        data_root=data_root,
        skip_categories=skip,
        seed=int(plan.get("seed") or SEED),
    )
    train_ex, val_ex, test_ex = _split_debug(examples)
    print(
        f"[DEBUG] Split: train={len(train_ex)} val={len(val_ex)} test={len(test_ex)}",
        flush=True,
    )

    exp_ids = resolve_experiment_ids(config.get("experiment_ids"))
    exp_id = int(args.exp_id or exp_ids[0])
    arts = ensure_cell_dirs(cell["id"], run_id=dest)
    src_aug = category_aug_path(cell["id"], run_id=source)
    dst_aug = category_aug_path(cell["id"], run_id=dest)
    if src_aug.is_file():
        shutil.copy(src_aug, dst_aug)
    else:
        proposed = plan.get("proposed_category_augmentation") or {"enabled": False, "categories": {}}
        dst_aug.write_text(json.dumps(proposed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    plan["split_uids"] = split_uids_from_example_lists(train_ex, val_ex, test_ex)
    plan["split_stats_by_category"] = {
        "train": _counts(train_ex),
        "val": _counts(val_ex),
        "test": _counts(test_ex),
    }
    plan["debug"] = {
        "n": len(examples),
        "epochs": int(args.epochs),
        "source_run_id": source,
        "seed": int(plan.get("seed") or SEED),
        "cell_id": cell["id"],
        "exp_id": exp_id,
    }
    plan.setdefault("campaign", {})["run_id"] = dest
    plan["category_augmentation_config"] = str(dst_aug.resolve())
    write_training_plan(plan, training_plan_path(cell["id"], run_id=dest))
    print(f"[DEBUG] Plan → {training_plan_path(cell['id'], run_id=dest)}", flush=True)

    os.environ["CAMPAIGN_DEBUG_EPOCHS"] = str(int(args.epochs))
    os.environ["PYTHONUNBUFFERED"] = "1"
    train_cell(
        cell,
        config,
        resume=False,
        exp_ids=[exp_id],
        run_id=dest,
        parallel=1,
        config_path=Path(args.config) if args.config else None,
        write_summary=True,
    )

    for split in ("val", "test"):
        print(f"\n[DEBUG] Evaluación split={split}", flush=True)
        evaluate_cell(cell, config, split=split, export_fp=False, run_id=dest)

    subprocess.check_call(
        [sys.executable, str(CAMPAIGN_DIR / "summarize_campaign.py"), "--run-id", dest, "--split", "val"]
    )
    print(f"[DEBUG] Hecho. Modelos de prueba en {arts['models_dir']}", flush=True)
    print(f"[DEBUG] El run {source} sigue intacto para el entrenamiento real.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
