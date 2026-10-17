"""Forward unificado (theft + concealment opcional)."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


def forward_theft(
    model: nn.Module,
    x: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    out = model(x, frame_mask)
    if isinstance(out, dict):
        return out["theft"]
    return out


def forward_heads(
    model: nn.Module,
    x: torch.Tensor,
    frame_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    out = model(x, frame_mask)
    if isinstance(out, dict):
        return out["theft"], out.get("concealment")
    return out, None
