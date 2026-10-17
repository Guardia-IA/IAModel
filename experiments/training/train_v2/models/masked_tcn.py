"""TCN dilated con pooling enmascarado (P1 baseline)."""
from __future__ import annotations

from typing import Any, Dict, Union

import torch
import torch.nn as nn


class _DilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm = nn.GroupNorm(8, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.norm(y)
        y = torch.relu(y)
        return x + self.drop(y)


def _masked_mean_max(h: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    m = frame_mask.unsqueeze(-1)
    denom = m.sum(dim=1).clamp(min=1.0)
    pooled_mean = (h * m).sum(dim=1) / denom
    neg_inf = torch.finfo(h.dtype).min
    h_fill = h.masked_fill(m <= 0, neg_inf)
    pooled_max = h_fill.max(dim=1).values
    pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
    return torch.cat([pooled_mean, pooled_max], dim=-1)


class MaskedTCNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.15,
        readout: str = "mean_max",
        aux_concealment: bool = False,
    ) -> None:
        super().__init__()
        self.readout = str(readout)
        self.aux_concealment = bool(aux_concealment)
        self.stem = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [_DilatedBlock(hidden_dim, dilation=2**i, dropout=dropout) for i in range(num_layers)]
        )
        feat_dim = hidden_dim * 2 if self.readout == "mean_max" else hidden_dim
        self.theft_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        if self.aux_concealment:
            self.conceal_head = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def _pool(self, h: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        if self.readout == "mean_max":
            return _masked_mean_max(h, frame_mask)
        m = frame_mask.unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1.0)
        return (h * m).sum(dim=1) / denom

    def forward(
        self, x: torch.Tensor, frame_mask: torch.Tensor | None = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        h = x.transpose(1, 2)
        h = self.stem(h)
        for blk in self.blocks:
            h = blk(h)
        h = h.transpose(1, 2)
        if frame_mask is None:
            frame_mask = torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)
        feat = self._pool(h, frame_mask)
        theft = self.theft_head(feat).squeeze(-1)
        if self.aux_concealment:
            conceal = self.conceal_head(feat).squeeze(-1)
            return {"theft": theft, "concealment": conceal}
        return theft
