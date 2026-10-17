"""Pérdidas MIL (bag-level log-mean-exp) + BCE ventana."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_indices(uids: List[str]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for i, u in enumerate(uids):
        groups.setdefault(str(u), []).append(i)
    return groups


def theft_train_loss(
    theft_logits: torch.Tensor,
    y_window: torch.Tensor,
    uids: List[str],
    clip_is_robbery: torch.Tensor,
    *,
    bag_loss_weight: float = 0.5,
    bag_temperature: float = 0.5,
    use_bag_for_robbery_clips: bool = True,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    BCE por ventana + opcional LSE pooling por clip de robo (experto §4).
    clip_is_robbery: [B] 1 si el clip es robo (manifest), independiente de etiqueta ventana MIL.
    """
    bce = nn.BCEWithLogitsLoss()
    window_loss = bce(theft_logits, y_window)
    parts = {"window_bce": float(window_loss.detach().item())}
    if not use_bag_for_robbery_clips or bag_loss_weight <= 0:
        return window_loss, parts

    bag_losses: List[torch.Tensor] = []
    groups = _group_indices(uids)
    for uid, idxs in groups.items():
        if not idxs:
            continue
        idx_t = torch.tensor(idxs, device=theft_logits.device, dtype=torch.long)
        if clip_is_robbery[idxs[0]].item() < 0.5:
            continue
        logits_g = theft_logits.index_select(0, idx_t)
        temp = max(float(bag_temperature), 1e-3)
        pooled = temp * torch.logsumexp(logits_g / temp, dim=0)
        target = torch.ones((), device=theft_logits.device, dtype=theft_logits.dtype)
        bag_losses.append(bce(pooled.unsqueeze(0), target.unsqueeze(0)))

    if bag_losses:
        bag_term = torch.stack(bag_losses).mean()
        parts["bag_bce"] = float(bag_term.detach().item())
        total = window_loss + float(bag_loss_weight) * bag_term
    else:
        parts["bag_bce"] = 0.0
        total = window_loss
    parts["total"] = float(total.detach().item())
    return total, parts
