#!/usr/bin/env python3
"""Entrenamiento v2 (un job Campaign2 P1)."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from train_v2.dataset_windows import WindowManifestDataset, make_datasets
from train_v2.metrics import binary_metrics, clip_metrics_from_window_probs
from train_v2.models.gru_pooled import GRUPooledClassifier
from train_v2.models.masked_tcn import MaskedTCNClassifier
from train_v2.window_index import WindowRef


def _build_model(arch: str, input_dim: int, cfg: Dict[str, Any]) -> nn.Module:
    hidden = int(cfg.get("hidden_dim", 64))
    dropout = float(cfg.get("dropout", 0.15))
    if arch in ("masked_tcn64", "masked_tcn"):
        return MaskedTCNClassifier(
            input_dim,
            hidden_dim=hidden,
            num_layers=int(cfg.get("num_layers", 4)),
            dropout=dropout,
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
) -> Dict[str, Any]:
    model.eval()
    probs: List[float] = []
    y_true: List[int] = []
    uids: List[str] = []
    idx = 0
    for x, y, m in loader:
        x = x.to(device)
        m = m.to(device)
        logits = model(x, m)
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
    clip_m = clip_metrics_from_window_probs(uids, yt, pr, threshold=threshold)
    return {"window": window_m, "clip_max": clip_m, "threshold": threshold}


def _train_sampler(ds: WindowManifestDataset) -> WeightedRandomSampler:
    labels = [ds.refs[i].label_theft for i in range(len(ds))]
    n_pos = max(sum(labels), 1)
    n_neg = max(len(labels) - n_pos, 1)
    w_pos = 0.5 / n_pos
    w_neg = 0.5 / n_neg
    weights = [w_pos if lb else w_neg for lb in labels]
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

    train_ds, val_ds, meta = make_datasets(
        manifest_path,
        temporal=temporal,
        robbery_class=robbery_class,
        max_windows_per_clip=max_wpc,
        min_valid_joint_ratio=min_valid,
    )
    input_dim = int(meta["input_dim"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=_train_sampler(train_ds),
        num_workers=int(params.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(params.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    model = _build_model(arch, input_dim, params).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loss_fn = nn.BCEWithLogitsLoss()

    best_score = -1.0
    best_path = output_dir / "best.pt"
    bad_epochs = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for x, y, m in train_loader:
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x, m)
            loss = loss_fn(logits, y)
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
        )
        clip = val_stats["clip_max"]
        score = float(clip["recall_pct"]) - 0.5 * float(clip["fp_rate_pct"])
        row = {
            "epoch": epoch,
            "train_loss": running / max(n_batches, 1),
            "val_clip_recall_pct": clip["recall_pct"],
            "val_clip_fp": clip["fp"],
            "val_clip_fp_rate_pct": clip["fp_rate_pct"],
            "val_window_f1_pct": val_stats["window"]["f1_pct"],
            "score": score,
        }
        history.append(row)

        if score > best_score:
            best_score = score
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "arch": arch,
                        "hidden_dim": params.get("hidden_dim", 64),
                        "dropout": params.get("dropout", 0.15),
                        "num_layers": params.get("num_layers", 4),
                        "train_v2": True,
                    },
                    "input_dim": input_dim,
                    "seq_len": meta["seq_len"],
                    "seed": seed,
                    "job_id": params.get("job_id"),
                    "preprocess": "campaign2_windows_legacy_norm_v1",
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
