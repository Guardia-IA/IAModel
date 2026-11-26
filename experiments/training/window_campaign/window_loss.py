"""B4 — weighted BCE training step for window campaign."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def train_one_epoch_weighted_bce(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    label_smoothing: float = 0.02,
    num_classes: int = 2,
    class_weight: Optional[torch.Tensor] = None,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        if len(batch) == 4:
            x, y, w, _mask = batch
        elif len(batch) == 3:
            x, y, w = batch
        else:
            x, y = batch
            w = torch.ones(y.shape[0], dtype=torch.float32)
        x = x.to(device)
        y = y.to(device).long()
        w = w.to(device).float()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if num_classes == 2:
            # CE sobre ambas clases (argmax en eval). BCE solo en logits[:,1] dejaba el modelo en "todo neg".
            loss_vec = F.cross_entropy(
                logits,
                y,
                reduction="none",
                weight=class_weight,
                label_smoothing=label_smoothing if label_smoothing > 0 else 0.0,
            )
        else:
            loss_vec = F.cross_entropy(
                logits, y, reduction="none", weight=class_weight
            )
        loss_vec = loss_vec * w
        denom = w.sum().clamp(min=1e-6)
        loss = loss_vec.sum() / denom
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * float(denom.item())
        n += int(denom.item())
    return total_loss / max(1, n)