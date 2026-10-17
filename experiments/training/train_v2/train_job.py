#!/usr/bin/env python3
"""Entrenamiento v2 (un job Campaign2 P1)."""
from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from train_v2.dataset_windows import WindowManifestDataset, collate_window_batch, make_datasets
from train_v2.mil_loss import theft_train_loss
from train_v2.model_forward import forward_heads, forward_theft
from train_v2.features import PREPROCESS_TORSO
from train_v2.metrics import (
    binary_metrics,
    clip_metrics_at_fp_budget,
    clip_metrics_from_window_probs,
)
from train_v2.models.gru_pooled import GRUPooledClassifier
from train_v2.models.masked_tcn import MaskedTCNClassifier
from train_v2.window_index import WindowRef


def _build_model(arch: str, input_dim: int, cfg: Dict[str, Any]) -> nn.Module:
    hidden = int(cfg.get("hidden_dim", 64))
    dropout = float(cfg.get("dropout", 0.15))
    ck = dict(cfg.get("config") or {})
    p1 = dict(cfg.get("p1_train") or ck or cfg)
    if arch in ("masked_tcn64", "masked_tcn"):
        return MaskedTCNClassifier(
            input_dim,
            hidden_dim=int(cfg.get("hidden_dim", ck.get("hidden_dim", hidden))),
            num_layers=int(cfg.get("num_layers", ck.get("num_layers", 4))),
            dropout=float(cfg.get("dropout", ck.get("dropout", dropout))),
            readout=str(p1.get("readout", ck.get("readout", "mean_max"))),
            aux_concealment=bool(
                p1.get("aux_concealment_head", ck.get("aux_concealment", True))
            ),
        )
    if arch in ("gru64_pooled", "gru_pooled"):
        return GRUPooledClassifier(
            input_dim,
            hidden_dim=hidden,
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=dropout,
        )
    raise ValueError(f"arch v2 no soportada: {arch}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _eval_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    refs: List[WindowRef],
    threshold: float,
    max_fp_clips: int = 11,
    clip_y_by_uid: Dict[str, int] | None = None,
) -> Dict[str, Any]:
    model.eval()
    probs: List[float] = []
    y_true: List[int] = []
    uids: List[str] = []
    idx = 0
    for batch in loader:
        if len(batch) == 5:
            x, y, m, _batch_uids, _crob = batch
        else:
            x, y, m = batch
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)
        logits = forward_theft(model, x, m)
        p = torch.sigmoid(logits).cpu().numpy()
        bs = x.size(0)
        for i in range(bs):
            probs.append(float(p[i]))
            y_true.append(int(y[i].item()))
            if idx < len(refs):
                uids.append(refs[idx].uid)
            idx += 1
    yt = np.array(y_true, dtype=np.int64)
    pr = np.array(probs, dtype=np.float64)
    pred = (pr >= threshold).astype(np.int64)
    window_m = binary_metrics(yt, pred)
    clip_m = clip_metrics_from_window_probs(
        uids, yt, pr, threshold=threshold, clip_y_by_uid=clip_y_by_uid
    )
    clip_budget = clip_metrics_at_fp_budget(
        uids, yt, pr, max_fp_clips=max_fp_clips, clip_y_by_uid=clip_y_by_uid
    )
    return {
        "window": window_m,
        "clip_max": clip_m,
        "clip_budget": clip_budget,
        "threshold": threshold,
    }


def _clip_labels_from_manifest(manifest: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for ent in manifest.get("entries") or []:
        uid = str(ent.get("uid") or "")
        if uid:
            out[uid] = 1 if ent.get("is_robbery") else 0
    return out


def _train_sampler(
    ds: WindowManifestDataset,
    *,
    hard_neg_classes: List[int] | None = None,
    hard_neg_manifest_uids: set[str] | None = None,
) -> WeightedRandomSampler:
    """25% pos / 25% hard neg / 50% neg ordinario (experto P3-lite)."""
    hard_neg_classes = hard_neg_classes or [2, 3, 4, 14]
    hard_set = {int(x) for x in hard_neg_classes}
    manifest_hard = hard_neg_manifest_uids or set()
    pos_idx: List[int] = []
    hard_idx: List[int] = []
    neg_idx: List[int] = []
    for i, ref in enumerate(ds.refs):
        if ref.label_theft:
            pos_idx.append(i)
        elif int(ref.clip_label) in hard_set or ref.uid in manifest_hard:
            hard_idx.append(i)
        else:
            neg_idx.append(i)
    n_pos = max(len(pos_idx), 1)
    n_hard = max(len(hard_idx), 1)
    n_neg = max(len(neg_idx), 1)
    w_pos = 0.25 / n_pos
    w_hard = 0.25 / n_hard
    w_neg = 0.50 / n_neg
    weights = [0.0] * len(ds.refs)
    for i in pos_idx:
        weights[i] = w_pos
    for i in hard_idx:
        weights[i] = w_hard
    for i in neg_idx:
        weights[i] = w_neg
    if sum(weights) <= 0:
        return WeightedRandomSampler([1.0] * len(ds), num_samples=len(ds), replacement=True)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def run_p1_train(params: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    t0 = time.perf_counter()
    seed = int(params.get("seed", 42))
    _set_seed(seed)

    manifest_path = Path(str(params["manifest_path"]))
    arch = str(params.get("arch", "masked_tcn64"))
    epochs = int(params.get("epochs", 100))
    batch_size = int(params.get("batch_size", 64))
    lr = float(params.get("lr", 3e-4))
    weight_decay = float(params.get("weight_decay", 0.01))
    grad_clip = float(params.get("grad_clip", 1.0))
    temporal = dict(params.get("temporal") or {})
    p1_cfg = dict(params.get("p1_train") or {})
    max_wpc = int(p1_cfg.get("max_windows_per_clip", 32))
    min_valid = float(p1_cfg.get("min_valid_joint_ratio", 0.25))
    val_threshold = float(p1_cfg.get("val_threshold", 0.5))
    early_patience = int(p1_cfg.get("early_stop_patience", 15))
    robbery_class = int(params.get("robbery_class", 6))
    selection_cfg = dict(params.get("selection") or {})
    max_fp_clips = int(
        p1_cfg.get("max_val_fp_clips")
        or selection_cfg.get("max_val_fp_clips")
        or 11
    )
    selection_metric = str(p1_cfg.get("selection_metric", "recall_at_fp"))
    warmup_epochs = int(p1_cfg.get("warmup_epochs", 5))
    normalize_mode = str(p1_cfg.get("preprocess", PREPROCESS_TORSO))
    augment_cfg = dict(p1_cfg.get("augment") or {})
    hard_neg_classes = [int(x) for x in (p1_cfg.get("hard_neg_classes") or [2, 3, 4, 14])]
    window_label_policy = str(p1_cfg.get("window_label_policy", "tail_positive"))
    robbery_tail_sec = float(p1_cfg.get("robbery_positive_tail_seconds", 3.0))
    hard_neg_manifest_path = params.get("hard_neg_manifest_path") or p1_cfg.get(
        "hard_neg_manifest_path"
    )
    hard_neg_uids: set[str] = set()
    if hard_neg_manifest_path:
        hn_path = Path(str(hard_neg_manifest_path))
        if hn_path.is_file():
            with open(hn_path, "r", encoding="utf-8") as hf:
                hard_neg_uids = {str(u) for u in (json.load(hf).get("uids") or [])}

    train_ds, val_ds, meta = make_datasets(
        manifest_path,
        temporal=temporal,
        robbery_class=robbery_class,
        max_windows_per_clip=max_wpc,
        min_valid_joint_ratio=min_valid,
        train_augment=bool(p1_cfg.get("augment_enabled", True)),
        augment_cfg=augment_cfg,
        normalize_mode=normalize_mode,
        augment_seed=seed,
        window_label_policy=window_label_policy,
        robbery_positive_tail_seconds=robbery_tail_sec,
    )
    clip_y_by_uid = _clip_labels_from_manifest(train_ds.manifest)
    input_dim = int(meta["input_dim"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=_train_sampler(
            train_ds,
            hard_neg_classes=hard_neg_classes,
            hard_neg_manifest_uids=hard_neg_uids,
        ),
        collate_fn=collate_window_batch,
        num_workers=int(params.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_window_batch,
        num_workers=int(params.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    model = _build_model(arch, input_dim, params).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_epochs = max(0, min(warmup_epochs, epochs - 1))
    if warmup_epochs > 0:
        warm = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(epochs - warmup_epochs, 1)
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warm, cosine], milestones=[warmup_epochs]
        )
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loss_fn = nn.BCEWithLogitsLoss()
    mil_bag_weight = float(p1_cfg.get("mil_bag_loss_weight", 0.5))
    mil_bag_temp = float(p1_cfg.get("mil_bag_temperature", 0.5))
    use_mil_bag = bool(p1_cfg.get("mil_bag_loss_enabled", True))
    conceal_weight = float(p1_cfg.get("concealment_loss_weight", 0.2))

    best_score = -1.0
    best_path = output_dir / "best.pt"
    bad_epochs = 0
    history: List[Dict[str, Any]] = []
    progress_path = output_dir / "progress.jsonl"

    def _log_progress(row: Dict[str, Any]) -> None:
        row = dict(row)
        row["job_id"] = params.get("job_id")
        row["epochs_total"] = epochs
        with open(progress_path, "a", encoding="utf-8") as pf:
            pf.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"[P1 {params.get('job_id')}] epoch {row.get('epoch')}/{epochs} "
            f"loss={row.get('train_loss', 0):.4f} "
            f"val_rec={row.get('val_clip_recall_pct', 0):.1f}% "
            f"val_FP={row.get('val_clip_fp')} score={row.get('score', 0):.2f} "
            f"thr={row.get('val_threshold', 0):.3f}",
            flush=True,
        )

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for x, y, m, uids, clip_rob in train_loader:
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            clip_rob = clip_rob.to(device)
            opt.zero_grad(set_to_none=True)
            theft_logits, conceal_logits = forward_heads(model, x, m)
            loss, _lp = theft_train_loss(
                theft_logits,
                y,
                uids,
                clip_rob,
                bag_loss_weight=mil_bag_weight,
                bag_temperature=mil_bag_temp,
                use_bag_for_robbery_clips=use_mil_bag,
            )
            if conceal_logits is not None and conceal_weight > 0:
                loss = loss + conceal_weight * loss_fn(conceal_logits, clip_rob)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            running += float(loss.item())
            n_batches += 1
        sched.step()

        val_stats = _eval_loader(
            model,
            val_loader,
            device,
            refs=val_ds.refs,
            threshold=val_threshold,
            max_fp_clips=max_fp_clips,
            clip_y_by_uid=clip_y_by_uid,
        )
        clip = val_stats["clip_max"]
        clip_budget = val_stats.get("clip_budget") or clip
        if selection_metric == "recall_at_fp":
            score = float(clip_budget.get("selection_score", clip_budget["recall_pct"]))
        else:
            score = float(clip["recall_pct"]) - 0.5 * float(clip["fp_rate_pct"])
        row = {
            "epoch": epoch,
            "train_loss": running / max(n_batches, 1),
            "val_clip_recall_pct": clip["recall_pct"],
            "val_clip_fp": clip["fp"],
            "val_clip_fp_rate_pct": clip["fp_rate_pct"],
            "val_clip_recall_at_fp_pct": clip_budget.get("recall_pct"),
            "val_clip_fp_at_budget": clip_budget.get("fp"),
            "val_threshold": clip_budget.get("threshold", val_threshold),
            "val_window_f1_pct": val_stats["window"]["f1_pct"],
            "score": score,
            "max_fp_clips": max_fp_clips,
        }
        history.append(row)
        _log_progress(row)

        if score > best_score:
            best_score = score
            bad_epochs = 0
            torch.save(
                {
                    "job_id": params.get("job_id"),
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "config": {
                        "arch": arch,
                        "hidden_dim": params.get("hidden_dim", 64),
                        "dropout": params.get("dropout", 0.15),
                        "num_layers": params.get("num_layers", 4),
                        "readout": p1_cfg.get("readout", "mean_max"),
                        "aux_concealment": bool(p1_cfg.get("aux_concealment_head", True)),
                        "train_v2": True,
                    },
                    "input_dim": input_dim,
                    "seq_len": meta["seq_len"],
                    "seed": seed,
                    "job_id": params.get("job_id"),
                    "preprocess": normalize_mode,
                    "selection": {
                        "metric": selection_metric,
                        "max_fp_clips": max_fp_clips,
                        "val_score": score,
                    },
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= early_patience:
                break

    wall = time.perf_counter() - t0
    result = {
        "job_type": "p1_train",
        "arch": arch,
        "seed": seed,
        "epochs_ran": len(history),
        "best_score": best_score,
        "best_checkpoint": str(best_path) if best_path.is_file() else None,
        "dataset": meta,
        "wall_seconds": wall,
        "last_epoch": history[-1] if history else None,
        "history_tail": history[-5:],
    }
    with open(output_dir / "p1_train_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return result
