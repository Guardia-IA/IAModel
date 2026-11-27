"""Single-experiment training loop for window_campaign2."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from campaign2_build_loaders import build_campaign2_datasets_and_loaders
from campaign2_metrics import recall_at_fpr_score
from campaign2_paths import load_campaign2_config

import sys

_ROOT = Path(__file__).resolve().parent
_TR = _ROOT.parent
_WC = _TR / "window_campaign"
for p in (_TR, _WC, _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from model_config import (  # type: ignore
    DEFAULT_BINARY_LOGIT_MARGIN,
    DEFAULT_BINARY_SOFTMAX_THRESHOLD,
    EXPERIMENTS,
)
from train_model_operations import (  # type: ignore
    AUGMENT_CONFIG_PATH,
    AUGMENT_MAX_OPS,
    AUGMENT_PROB,
    AUGMENT_PROFILE_DEFAULT,
    MANIFEST_VARIANT_SET_DEFAULT,
    MAX_DETERMINISTIC_VARIANTS,
    SEED,
    TRAIN_DETERMINISTIC_PROB,
    build_model,
    evaluate_with_metrics,
    format_binary_metrics_line,
)
from window_loss import train_one_epoch_weighted_bce  # type: ignore
from window_parallel_policy import resolve_dataloader_workers  # type: ignore
from window_train_progress import set_experiment_running, tick_epoch  # type: ignore


def _aug_from_campaign(cfg: Dict[str, Any]) -> Dict[str, Any]:
    camp = cfg.get("_campaign_config") or {}
    syn = cfg.get("synthetic_variants") or {}
    profile_id = syn.get("aug_profile", "baseline")
    profiles = camp.get("aug_profiles") or {}
    prof = profiles.get(profile_id, {})
    train_opts = dict(prof.get("train_opts") or {})
    train_opts.setdefault("augment_profile", profile_id)
    return train_opts


def _checkpoint_score(
    *,
    task: str,
    val_acc: float,
    val_metrics: Dict[str, Any],
    metric_key: str,
    cfg: Dict[str, Any],
) -> float:
    td = cfg.get("train_defaults") or {}
    if metric_key == "recall_at_fpr_cap" and task == "binary":
        return recall_at_fpr_score(
            val_metrics,
            float(td.get("target_fpr_pct", 1.0)),
            float(td.get("min_recall_robbery_pct", 66.0)),
        )
    if metric_key == "macro_f1":
        f1 = float(val_metrics.get("macro_f1", 0.0))
        if task == "binary":
            block = (val_metrics.get("binary") or {}).get("softmax_argmax") or {}
            if float(block.get("recall_robbery_pct", 0.0)) <= 0.0:
                return -1.0
        return f1
    return float(val_metrics.get("macro_f1", val_acc))


def run_campaign2_experiment(
    exp_id: int,
    *,
    data_root: Path,
    device: torch.device,
    cfg: Dict[str, Any],
    augment_on_the_fly: bool,
    manifest_cache_dir: Optional[Path],
    models_out: Path,
    progress_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if exp_id < 1 or exp_id > len(EXPERIMENTS):
        raise ValueError(f"experiment-id {exp_id} out of range")

    exp_cfg = dict(EXPERIMENTS[exp_id - 1])
    aug_opts = _aug_from_campaign(cfg)
    syn = cfg.get("synthetic_variants") or {}
    train_def = cfg.get("train_defaults") or {}

    window_len = int(cfg.get("window_len", 96))
    seq_len = window_len
    exp_cfg["seq_len"] = window_len

    batch_size = int(train_def.get("batch_size", exp_cfg.get("batch_size", 32)))
    epochs = int(train_def.get("epochs", exp_cfg.get("epochs", 25)))
    lr = float(exp_cfg.get("lr", 1e-3))
    pose_source = str(cfg.get("pose_source", "full"))
    augment_profile = str(aug_opts.get("augment_profile", AUGMENT_PROFILE_DEFAULT))
    debug_clips = int(cfg.get("_debug_max_clips") or 0)

    par = cfg.get("parallelism") or {}
    import os

    if os.environ.get("WINDOW_DATALOADER_WORKERS"):
        num_workers = int(os.environ["WINDOW_DATALOADER_WORKERS"])
    else:
        num_workers = resolve_dataloader_workers(par, os.cpu_count() or 8, 1)

    loaders, input_dim, label_to_idx, summary = build_campaign2_datasets_and_loaders(
        data_root=data_root,
        cfg=cfg,
        batch_size=batch_size,
        num_workers=num_workers,
        debug_max_clips=debug_clips,
        augment_on_the_fly=augment_on_the_fly,
        augment_config_path=AUGMENT_CONFIG_PATH,
        augment_profile=augment_profile,
        augment_prob=float(aug_opts.get("augment_prob", AUGMENT_PROB)),
        augment_max_ops=int(aug_opts.get("augment_max_ops", AUGMENT_MAX_OPS)),
        augment_seed=SEED,
        max_deterministic_variants=int(syn.get("max_deterministic_variants", MAX_DETERMINISTIC_VARIANTS)),
        train_deterministic_prob=float(syn.get("train_deterministic_prob", TRAIN_DETERMINISTIC_PROB)),
        use_deterministic_in_train=bool(syn.get("use_deterministic_in_train", True)),
        manifest_cache_dir=manifest_cache_dir,
        manifest_variant_set=MANIFEST_VARIANT_SET_DEFAULT,
        extra_manifest_views_per_window=int(syn.get("extra_manifest_views_per_clip", 1)),
        force_new_split=bool(cfg.get("_force_new_split")),
        cell_id=cfg.get("default_cell_id"),
    )

    task = "binary"
    num_classes = len(label_to_idx)
    model = build_model(exp_cfg["arch"], input_dim, num_classes, exp_cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=float(exp_cfg.get("weight_decay", 0.0))
    )

    loss_cfg = cfg.get("loss") or {}
    td = cfg.get("train_defaults") or {}
    ckpt_metric = str(td.get("checkpoint_metric") or "recall_at_fpr_cap")
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.02))
    auto_pos_w = bool(loss_cfg.get("auto_pos_class_weight", False))
    class_weight = None

    print(
        f"\n{'=' * 72}\n[WIN2] exp {exp_id:02d} | arch={exp_cfg.get('arch')} | epochs={epochs} | "
        f"checkpoint={ckpt_metric}\n{'=' * 72}",
        flush=True,
    )
    if progress_path:
        set_experiment_running(progress_path, exp_id, epochs_total=epochs, arch=str(exp_cfg.get("arch")))

    best_score = float("-inf")
    best_state = None
    train_sampler = getattr(loaders["train"], "sampler", None)

    eval_kwargs = dict(
        task=task,
        binary_softmax_threshold=DEFAULT_BINARY_SOFTMAX_THRESHOLD,
        binary_logit_margin=DEFAULT_BINARY_LOGIT_MARGIN,
    )

    for epoch in range(1, epochs + 1):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch_weighted_bce(
            model,
            loaders["train"],
            optimizer,
            device,
            label_smoothing=label_smoothing,
            num_classes=num_classes,
            class_weight=class_weight,
        )
        val_loss, val_acc, val_metrics = evaluate_with_metrics(
            model, loaders["val"], nn.CrossEntropyLoss(), device, num_classes=num_classes, **eval_kwargs
        )
        line = f"[WIN2 {exp_id:02d}] ep {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        if val_metrics.get("binary"):
            line += " | " + format_binary_metrics_line(val_metrics["binary"])
        print(line, flush=True)
        ckpt = _checkpoint_score(
            task=task, val_acc=val_acc, val_metrics=val_metrics, metric_key=ckpt_metric, cfg=cfg
        )
        if progress_path:
            tick_epoch(
                progress_path,
                exp_id,
                epoch,
                extra={"val_acc": round(float(val_acc), 4), "checkpoint_score": round(float(ckpt), 4)},
            )
        if ckpt > best_score:
            best_score = ckpt
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_metrics = evaluate_with_metrics(
        model, loaders["test"], nn.CrossEntropyLoss(), device, num_classes=num_classes, **eval_kwargs
    )
    print(
        f"[WIN2 {exp_id:02d}] TEST acc={test_acc:.4f} macro_f1={test_metrics.get('macro_f1', 0):.4f}"
    )
    if test_metrics.get("binary"):
        print(format_binary_metrics_line(test_metrics["binary"], prefix="test: "))

    models_out.mkdir(parents=True, exist_ok=True)
    out_path = models_out / f"modelo_{exp_id:02d}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": exp_cfg["arch"],
            "experiment_id": exp_id,
            "input_dim": input_dim,
            "seq_len": window_len,
            "label_to_idx": label_to_idx,
            "cell_id": cfg.get("default_cell_id"),
            "campaign2_config": "win2_native",
        },
        out_path,
    )
    print(f"[WIN2 {exp_id:02d}] Guardado {out_path}", flush=True)
    return {
        "exp_id": exp_id,
        "best_checkpoint_score": best_score,
        "test_metrics": test_metrics,
        "summary": summary,
    }