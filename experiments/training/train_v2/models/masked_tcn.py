"""TCN dilated con pooling enmascarado (P1 baseline)."""
from __future__ import annotations

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


class MaskedTCNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [_DilatedBlock(hidden_dim, dilation=2**i, dropout=dropout) for i in range(num_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, F]
        h = x.transpose(1, 2)
        h = self.stem(h)
        for blk in self.blocks:
            h = blk(h)
        h = h.transpose(1, 2)
        if frame_mask is None:
            frame_mask = torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)
        m = frame_mask.unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1.0)
        pooled = (h * m).sum(dim=1) / denom
        return self.head(pooled).squeeze(-1)
