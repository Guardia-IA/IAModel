#!/usr/bin/env python3
"""
Entrena los experimentos seleccionados para cada celda de la campaña.

Uso:
  python train_campaign.py --cells bin_full
  python train_campaign.py --all
  python train_campaign.py --all --resume
  python train_campaign.py --run-id improve_v1 --config campaign_config_improve.json --all
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
        load_merged_campaign_config,
        filter_cells,
        ensure_cell_dirs,
        training_plan_path,
        category_aug_path,
        hard_negative_manifest_path,
        resolve_experiment_ids,
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
    return dict((config.get("aug_profiles") or {})[profile_id])


def _train_opts_for_cell(cell: Dict[str, Any], config: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    aug_id = plan.get("campaign", {}).get("aug_profile") or cell.get("aug_profile")
    if not aug_id and cell.get("improve_profile"):
        ip = (config.get("improve_profiles") or {})[cell["improve_profile"]]
        aug_id = ip.get("base_aug_profile", "fp_hardened")
    aug_prof = _aug_profile(config, str(aug_id))
    train_opts = dict(aug_prof.get("train_opts") or {})
    improve = plan.get("campaign", {}).get("improve") or {}
    ip_id = improve.get("improve_profile") or cell.get("improve_profile")
    if ip_id:
        ip = (config.get("improve_profiles") or {}).get(ip_id, {})
        train_opts.update(ip.get("train_opts") or {})
    return train_opts


def _model_exists(models_dir: Path, exp_id: int) -> bool:
    return (models_dir / f"modelo_{exp_id:02d}.pt").is_file()


def train_cell(
    cell: Dict[str, Any],
    config: Dict[str, Any],
    *,
    resume: bool = False,
    exp_ids: Optional[List[int]] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    cell_id = cell["id"]
    arts = ensure_cell_dirs(cell_id, run_id=run_id)
    models_dir = arts["models_dir"]
    splits_dir = arts["splits_dir"]
    plan_path = training_plan_path(cell_id, run_id=run_id)
    if not plan_path.is_file():
        raise FileNotFoundError(f"Falta plan: {plan_path}. Ejecuta preflight_campaign.py --write-all")

    plan = load_training_plan_json(plan_path)
    if exp_ids is not None:
        exp_list = resolve_experiment_ids(exp_ids)
    else:
        from learning_curve_utils import experiment_ids_for_cell

        exp_list = experiment_ids_for_cell(cell, config)
    train_opts = _train_opts_for_cell(cell, config, plan)

    hn_manifest: Optional[Path] = None
    hn_weight = 3.0
    hn_path_str = plan.get("campaign", {}).get("hard_negative_manifest")
    if hn_path_str and Path(hn_path_str).is_file():
        hn_manifest = Path(hn_path_str)
    elif hard_negative_manifest_path(cell_id, run_id=run_id).is_file():
        hn_manifest = hard_negative_manifest_path(cell_id, run_id=run_id)
    if hn_manifest:
        with open(hn_manifest, "r", encoding="utf-8") as f:
            hn_data = json.load(f)
        hn_weight = float(hn_data.get("uid_weight", hn_weight))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs_thr = float(plan.get("evaluation", {}).get("binary_softmax_threshold", DEFAULT_BINARY_SOFTMAX_THRESHOLD))
    bs_margin = float(plan.get("evaluation", {}).get("binary_logit_margin", DEFAULT_BINARY_LOGIT_MARGIN))

    results: List[Dict[str, Any]] = []
    wall_total = 0.0

    print(f"\n{'=' * 80}\nCelda {cell_id} | task={cell['task']} pose={cell['pose_source']}")
    if run_id:
        print(f"Run: {run_id}")
    print(f"{'=' * 80}")
    print(f"Plan: {plan_path}")
    print(f"Modelos: {models_dir}")
    if hn_manifest:
        print(f"Hard negatives: {hn_manifest}")
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
                category_aug_config_path=category_aug_path(cell_id, run_id=run_id),
                training_plan_path=plan_path,
                loss_type=str(train_opts.get("loss_type", "ce")),
                hard_negative_mining=bool(train_opts.get("hard_negative_mining", False)),
                target_neg_pos_ratio=train_opts.get("target_neg_pos_ratio"),
                maintain_class_ratio=bool(train_opts.get("maintain_class_ratio", False)),
                hard_negative_uid_manifest=hn_manifest,
                hard_negative_uid_weight=hn_weight,
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
            res["run_id"] = run_id
            results.append(res)
            wall_total += time.perf_counter() - t0
        except Exception as exc:
            print(f"[Exp {exp_id:02d}] ERROR: {exc}")
            results.append({"exp_id": exp_id, "cell_id": cell_id, "run_id": run_id, "error": str(exc)})

    summary_path = arts["reports_dir"] / "train_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cell_id": cell_id,
        "run_id": run_id,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "wall_seconds": wall_total,
        "experiment_ids": exp_list,
        "hard_negative_manifest": str(hn_manifest) if hn_manifest else None,
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
    ap.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="ID de run (artefactos en artifacts/runs/<run-id>/)",
    )
    ap.add_argument(
        "--learning-curve",
        "--prediction",
        dest="learning_curve",
        action="store_true",
        help="Entrena cada tamaño de la curva (run_id lc_<N>)",
    )
    ap.add_argument(
        "--train-sizes",
        nargs="+",
        default=None,
        metavar="N|max",
        help="Tamaños train (entero o 'max'; sin flag → manifiesto del preflight)",
    )
    args = ap.parse_args()

    config = load_merged_campaign_config(Path(args.config) if args.config else None)

    if args.learning_curve:
        from learning_curve_utils import (
            get_learning_curve_train_sizes,
            resolve_learning_curve_cells,
            run_id_for_train_size,
            experiment_ids_for_cell,
        )

        try:
            cells = resolve_learning_curve_cells(config, args.cells)
            train_sizes = get_learning_curve_train_sizes(
                cli_sizes=args.train_sizes,
                config=config,
            )
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

        for cell in cells:
            cid = cell["id"]
            exp_ids = args.exp_ids or experiment_ids_for_cell(cell, config)
            for n in train_sizes:
                rid = run_id_for_train_size(n)
                print(f"\n{'#' * 80}\n# LC train — {cid} size={n} run_id={rid}\n{'#' * 80}")
                train_cell(cell, config, resume=args.resume, exp_ids=exp_ids, run_id=rid)
        return 0

    run_id = str(args.run_id).strip() if args.run_id else None
    if args.all or not args.cells:
        cells = filter_cells(config, None)
    else:
        cells = filter_cells(config, args.cells)

    if not cells:
        print("Indica --all o --cells id1 id2 ...", file=sys.stderr)
        return 1

    for cell in cells:
        cell_exp_ids = args.exp_ids or None
        train_cell(cell, config, resume=args.resume, exp_ids=cell_exp_ids, run_id=run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
