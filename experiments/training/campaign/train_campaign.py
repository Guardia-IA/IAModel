#!/usr/bin/env python3
"""
Entrena los experimentos seleccionados para cada celda de la campaña.

Uso:
  python train_campaign.py --cells bin_full
  python train_campaign.py --all
  python train_campaign.py --all --resume
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
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
        load_campaign_config,
        filter_cells,
        ensure_cell_dirs,
        training_plan_path,
        category_aug_path,
    )
    from model_config import EXPERIMENTS, DEFAULT_BINARY_SOFTMAX_THRESHOLD, DEFAULT_BINARY_LOGIT_MARGIN
    from train_model_operations import (
        run_experiment,
        load_training_plan_json,
        AUGMENT_CONFIG_PATH,
        AUGMENT_PROFILE_DEFAULT,
        AUGMENT_PROB,
        AUGMENT_MAX_OPS,
        SEED,
        MAX_DETERMINISTIC_VARIANTS,
        TRAIN_DETERMINISTIC_PROB,
        MIN_CLIP_SECONDS,
        MIN_VALID_FRAMES,
        MIN_VALID_PCT,
        MAX_OCCLUSION_RATIO,
    )
except ImportError as exc:
    raise SystemExit(f"Import error (¿entorno con torch?): {exc}") from exc


def _aug_profile(config: Dict[str, Any], profile_id: str) -> Dict[str, Any]:
    return (config.get("aug_profiles") or {})[profile_id]


def _model_exists(models_dir: Path, exp_id: int) -> bool:
    return (models_dir / f"modelo_{exp_id:02d}.pt").is_file()


def train_cell(
    cell: Dict[str, Any],
    config: Dict[str, Any],
    *,
    resume: bool = False,
    exp_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    cell_id = cell["id"]
    arts = ensure_cell_dirs(cell_id)
    models_dir = arts["models_dir"]
    splits_dir = arts["splits_dir"]
    plan_path = training_plan_path(cell_id)
    if not plan_path.is_file():
        raise FileNotFoundError(f"Falta plan: {plan_path}. Ejecuta preflight_campaign.py --write-all")

    plan = load_training_plan_json(plan_path)
    exp_list = exp_ids or list(config.get("experiment_ids", []))
    aug_prof = _aug_profile(config, cell["aug_profile"])
    train_opts = dict(aug_prof.get("train_opts") or {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs_thr = float(plan.get("evaluation", {}).get("binary_softmax_threshold", DEFAULT_BINARY_SOFTMAX_THRESHOLD))
    bs_margin = float(plan.get("evaluation", {}).get("binary_logit_margin", DEFAULT_BINARY_LOGIT_MARGIN))

    results: List[Dict[str, Any]] = []
    wall_total = 0.0

    print(f"\n{'=' * 80}\nCelda {cell_id} | task={cell['task']} pose={cell['pose_source']}\n{'=' * 80}")
    print(f"Plan: {plan_path}")
    print(f"Modelos: {models_dir}")
    print(f"Experimentos: {exp_list}")

    for exp_id in exp_list:
        if exp_id < 1 or exp_id > len(EXPERIMENTS):
            print(f"[Exp {exp_id:02d}] Fuera de rango EXPERIMENTS, omitido.")
            continue
        if resume and _model_exists(models_dir, exp_id):
            print(f"[Exp {exp_id:02d}] Ya existe, resume → omitido.")
            continue

        cfg = dict(EXPERIMENTS[exp_id - 1])
        t0 = time.perf_counter()
        try:
            res = run_experiment(
                exp_id,
                cfg,
                device,
                task=cell["task"],
                positive_class=int(config.get("robbery_class", 6)),
                pose_source_override=cell["pose_source"],
                single_user_only=bool(config.get("single_user_only", True)),
                models_dir=models_dir,
                min_clip_seconds=MIN_CLIP_SECONDS,
                min_valid_frames=MIN_VALID_FRAMES,
                min_valid_pct=MIN_VALID_PCT,
                max_occlusion_ratio=MAX_OCCLUSION_RATIO,
                split_manifest_out=(splits_dir / f"split_manifest_exp_{exp_id:02d}.json"),
                category_aug_config_path=category_aug_path(cell_id),
                training_plan_path=plan_path,
                loss_type=str(train_opts.get("loss_type", "ce")),
                hard_negative_mining=bool(train_opts.get("hard_negative_mining", False)),
                target_neg_pos_ratio=train_opts.get("target_neg_pos_ratio"),
                maintain_class_ratio=bool(train_opts.get("maintain_class_ratio", False)),
                augment_config_path=AUGMENT_CONFIG_PATH,
                augment_profile=AUGMENT_PROFILE_DEFAULT,
                augment_prob=AUGMENT_PROB,
                augment_max_ops=AUGMENT_MAX_OPS,
                augment_seed=SEED,
                max_deterministic_variants=MAX_DETERMINISTIC_VARIANTS,
                train_deterministic_prob=TRAIN_DETERMINISTIC_PROB,
                binary_softmax_threshold=bs_thr,
                binary_logit_margin=bs_margin,
            )
            res["cell_id"] = cell_id
            results.append(res)
            wall_total += time.perf_counter() - t0
        except Exception as exc:
            print(f"[Exp {exp_id:02d}] ERROR: {exc}")
            results.append({"exp_id": exp_id, "cell_id": cell_id, "error": str(exc)})

    summary_path = arts["reports_dir"] / "train_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cell_id": cell_id,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "wall_seconds": wall_total,
        "experiment_ids": exp_list,
        "results": results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\nResumen train → {summary_path} ({wall_total / 60:.1f} min)")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Entrenamiento por celdas de campaña")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="Todas las celdas del config")
    ap.add_argument("--resume", action="store_true", help="Omite modelos ya entrenados")
    ap.add_argument("--exp-ids", type=int, nargs="*", default=None, help="Override experiment_ids")
    args = ap.parse_args()

    config = load_campaign_config(Path(args.config) if args.config else None)
    if args.all or not args.cells:
        cells = filter_cells(config, None)
    else:
        cells = filter_cells(config, args.cells)

    if not cells:
        print("Indica --all o --cells id1 id2 ...", file=sys.stderr)
        return 1

    for cell in cells:
        train_cell(cell, config, resume=args.resume, exp_ids=args.exp_ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
