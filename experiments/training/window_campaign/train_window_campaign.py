#!/usr/bin/env python3
"""
Entrena experimentos por ventanas 8s (sin tocar train_campaign.py).

  python train_window_campaign.py --data-root /path/data_result --experiment-id 48
  python train_window_campaign.py --experiment-id 48 49 --no-augment
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

WINDOW_DIR = Path(__file__).resolve().parent
TRAINING_DIR = WINDOW_DIR.parent
for p in (TRAINING_DIR, WINDOW_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from model_config import (  # type: ignore
    EXPERIMENTS,
    DEFAULT_BINARY_LOGIT_MARGIN,
    DEFAULT_BINARY_SOFTMAX_THRESHOLD,
)
from train_model_operations import (  # type: ignore
    AUGMENT_CONFIG_PATH,
    AUGMENT_MAX_OPS,
    AUGMENT_PROFILE_DEFAULT,
    AUGMENT_PROB,
    MANIFEST_VARIANT_SET_DEFAULT,
    MAX_DETERMINISTIC_VARIANTS,
    SEED,
    TRAIN_DETERMINISTIC_PROB,
    build_model,
    evaluate_with_metrics,
    format_binary_metrics_line,
    get_data_result_roots,
    positive_class_set,
)
from window_build_loaders import build_window_datasets_and_loaders
from window_loss import train_one_epoch_weighted_bce
from window_paths import load_window_config, models_dir


def _aug_from_campaign(cfg_window: Dict[str, Any]) -> Dict[str, Any]:
    camp = cfg_window.get("_campaign_config") or {}
    syn = cfg_window.get("synthetic_variants") or {}
    profile_id = syn.get("aug_profile", "fp_hardened")
    profiles = camp.get("aug_profiles") or {}
    prof = profiles.get(profile_id, {})
    train_opts = dict(prof.get("train_opts") or {})
    train_opts.setdefault("augment_profile", profile_id)
    return train_opts


def run_window_experiment(
    exp_id: int,
    *,
    data_root: Path,
    device: torch.device,
    cfg_window: Dict[str, Any],
    task: str,
    positive_classes: List[int],
    augment_on_the_fly: bool,
    manifest_cache_dir: Optional[Path],
    models_out: Path,
) -> Dict[str, Any]:
    if exp_id < 1 or exp_id > len(EXPERIMENTS):
        raise ValueError(f"experiment-id {exp_id} fuera de rango 1..{len(EXPERIMENTS)}")

    exp_cfg = dict(EXPERIMENTS[exp_id - 1])
    aug_opts = _aug_from_campaign(cfg_window)
    syn = cfg_window.get("synthetic_variants") or {}
    split = cfg_window.get("split") or {}
    train_def = cfg_window.get("train_defaults") or {}

    seq_len = int(exp_cfg.get("seq_len", cfg_window.get("window_len", 96)))
    window_len = int(cfg_window.get("window_len", 96))
    if seq_len != window_len:
        print(
            f"[AVISO] exp seq_len={seq_len} != window_len={window_len}; "
            f"se usa window_len={window_len} para cortes y seq_len del modelo={window_len}"
        )
        seq_len = window_len
        exp_cfg["seq_len"] = window_len

    batch_size = int(train_def.get("batch_size", exp_cfg.get("batch_size", 32)))
    epochs = int(train_def.get("epochs", exp_cfg.get("epochs", 25)))
    lr = float(exp_cfg.get("lr", 1e-3))
    pose_source = str(cfg_window.get("pose_source", exp_cfg.get("pose_source", "full")))
    augment_profile = str(aug_opts.get("augment_profile", AUGMENT_PROFILE_DEFAULT))

    debug_clips = int(cfg_window.get("_debug_max_clips") or 0)

    loaders, input_dim, label_to_idx, summary = build_window_datasets_and_loaders(
        data_root=data_root,
        cfg_window=cfg_window,
        batch_size=batch_size,
        debug_max_clips=debug_clips,
        augment_on_the_fly=augment_on_the_fly,
        augment_config_path=AUGMENT_CONFIG_PATH,
        augment_profile=augment_profile,
        augment_prob=float(aug_opts.get("augment_prob", AUGMENT_PROB)),
        augment_max_ops=int(aug_opts.get("augment_max_ops", AUGMENT_MAX_OPS)),
        augment_seed=SEED,
        max_deterministic_variants=int(
            syn.get("max_deterministic_variants", MAX_DETERMINISTIC_VARIANTS)
        ),
        train_deterministic_prob=float(
            syn.get("train_deterministic_prob", TRAIN_DETERMINISTIC_PROB)
        ),
        use_deterministic_in_train=bool(syn.get("use_deterministic_in_train", True)),
        manifest_cache_dir=manifest_cache_dir,
        manifest_variant_set=MANIFEST_VARIANT_SET_DEFAULT,
        extra_manifest_views_per_window=int(syn.get("extra_manifest_views_per_clip", 1)),
        force_new_split=bool(cfg_window.get("_force_new_split")),
        cell_id=cfg_window.get("default_cell_id"),
    )
    loss_cfg = cfg_window.get("loss") or {}
    num_classes = len(label_to_idx)
    model = build_model(exp_cfg["arch"], input_dim, num_classes, exp_cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=float(exp_cfg.get("weight_decay", 0.0))
    )

    eval_kwargs = dict(
        task=task,
        binary_softmax_threshold=DEFAULT_BINARY_SOFTMAX_THRESHOLD,
        binary_logit_margin=DEFAULT_BINARY_LOGIT_MARGIN,
    )

    best_val = 0.0
    best_state = None
    t0 = time.perf_counter()
    print(f"\n{'=' * 72}\nWindow exp {exp_id:02d} | arch={exp_cfg.get('arch')} | epochs={epochs}\n{'=' * 72}")

    label_smoothing = float(loss_cfg.get("label_smoothing", 0.02))
    for epoch in range(1, epochs + 1):
        if str(loss_cfg.get("type", "weighted_bce")) == "weighted_bce":
            train_loss = train_one_epoch_weighted_bce(
                model,
                loaders["train"],
                optimizer,
                device,
                label_smoothing=label_smoothing,
                num_classes=num_classes,
            )
        else:
            from train_model_operations import train_one_epoch

            train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc, val_metrics = evaluate_with_metrics(
            model, loaders["val"], criterion, device, num_classes=num_classes, **eval_kwargs
        )
        line = f"[Win {exp_id:02d}] ep {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        if task == "binary" and val_metrics.get("binary"):
            line += " | " + format_binary_metrics_line(val_metrics["binary"])
        print(line)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_metrics = evaluate_with_metrics(
        model, loaders["test"], criterion, device, num_classes=num_classes, **eval_kwargs
    )
    print(
        f"[Win {exp_id:02d}] TEST loss={test_loss:.4f} acc={test_acc:.4f} "
        f"macro_f1={test_metrics.get('macro_f1', 0):.4f}"
    )
    if task == "binary" and test_metrics.get("binary"):
        print(format_binary_metrics_line(test_metrics["binary"], prefix="test: "))

    models_out.mkdir(parents=True, exist_ok=True)
    run_config = {
        "experiment_id": exp_id,
        "architecture": exp_cfg.get("arch"),
        "cell_id": cfg_window.get("default_cell_id"),
        "loss_type": loss_cfg.get("type", "weighted_bce"),
        "label_mode": (cfg_window.get("positive_labeling") or {}).get("mode"),
        "mask_policy": cfg_window.get("mask_policy"),
        "preprocess_version": cfg_window.get("preprocess_version"),
        "sampler": cfg_window.get("sampler"),
        "synthetic_variants": syn,
        "padding": cfg_window.get("padding"),
        "window_len": window_len,
        "stride": cfg_window.get("stride", 48),
    }
    run_config_path = models_out / f"run_config_exp_{exp_id:02d}.json"
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    ckpt_path = models_out / f"modelo_{exp_id:02d}.pt"
    checkpoint = {
        "state_dict": model.state_dict(),
        "exp_id": exp_id,
        "arch": exp_cfg.get("arch"),
        "seq_len": window_len,
        "input_dim": input_dim,
        "label_to_idx": label_to_idx,
        "task": task,
        "positive_classes": sorted(positive_class_set(6, positive_classes)),
        "window_campaign": True,
        "window_campaign_r2": True,
        "run_config": run_config,
        "window_len": window_len,
        "stride": cfg_window.get("stride", 48),
        "data_root": str(data_root),
        "augment_profile": augment_profile,
        "summary": summary,
        "test_metrics": test_metrics,
        "best_val_acc": best_val,
        "train_seconds": time.perf_counter() - t0,
    }
    torch.save(checkpoint, ckpt_path)
    meta_path = models_out / f"window_train_meta_{exp_id:02d}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in checkpoint.items() if k != "state_dict"}, f, indent=2)
    print(f"Checkpoint: {ckpt_path}")
    return checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--experiment-id", type=int, nargs="*", default=None)
    ap.add_argument(
        "--stage",
        choices=("pilot", "stage2", "stage3", "benchmark"),
        default="benchmark",
        help="benchmark=6 arq (48,49,53,57,13,11) | pilot | stage2 | stage3",
    )
    ap.add_argument("--debug-clips", type=int, default=0, help="Entrenar solo con N clips (→ N×chunks)")
    ap.add_argument("--debug-epochs", type=int, default=0, help="Si >0, sustituye épocas del config")
    ap.add_argument("--cell-id", type=str, default=None)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument(
        "--force-new-split",
        action="store_true",
        help="Ignora manifiesto congelado y regenera train/val/test",
    )
    ap.add_argument(
        "--manifest-cache-dir",
        type=str,
        default=None,
        help="Caché JSON validate_npy (misma que campaña)",
    )
    args = ap.parse_args()

    cfg = load_window_config()
    cell_id = args.cell_id or cfg.get("default_cell_id", "win_bin_full_baseline")
    stages = cfg.get("experiment_stages") or {}
    if args.experiment_id:
        exp_ids = list(args.experiment_id)
    else:
        key_map = {
            "pilot": "stage1_pilot",
            "stage2": "stage2",
            "stage3": "stage3_optional",
            "benchmark": "benchmark",
        }
        key = key_map[args.stage]
        exp_ids = list(stages.get(key) or cfg.get("experiment_benchmark", [48, 49]))
    data_root = (
        Path(args.data_root).expanduser().resolve()
        if args.data_root
        else get_data_result_roots(None)[0]
    )
    task = "binary"
    if args.force_new_split:
        cfg = dict(cfg)
        cfg["_force_new_split"] = True
    if args.debug_clips > 0 or args.debug_epochs > 0:
        cfg = dict(cfg) if not args.force_new_split else cfg
        if args.debug_clips > 0:
            cfg["_debug_max_clips"] = args.debug_clips
        if args.debug_epochs > 0:
            td = dict(cfg.get("train_defaults") or {})
            td["epochs"] = args.debug_epochs
            cfg["train_defaults"] = td
    positive = cfg.get("robbery_classes", [6, 14])
    augment = not args.no_augment and bool(cfg.get("synthetic_variants", {}).get("augment_on_the_fly", True))
    mdir = Path(args.manifest_cache_dir).resolve() if args.manifest_cache_dir else None
    if mdir is None and cfg.get("synthetic_variants", {}).get("use_validate_npy_manifests"):
        # Ruta típica campaña (si existe)
        guess = TRAINING_DIR / "operations_npy" / "manifest_cache"
        mdir = guess if guess.is_dir() else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = models_dir(cell_id)

    print(f"[WINDOW] stage={args.stage} experiment_ids={exp_ids} cell={cell_id}")
    for eid in exp_ids:
        run_window_experiment(
            eid,
            data_root=data_root,
            device=device,
            cfg_window=cfg,
            task=task,
            positive_classes=positive,
            augment_on_the_fly=augment,
            manifest_cache_dir=mdir,
            models_out=out,
        )


if __name__ == "__main__":
    main()