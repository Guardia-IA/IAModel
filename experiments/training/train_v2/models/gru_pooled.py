"""BiGRU + pooling enmascarado."""
from __future__ import annotations

import torch
import torch.nn as nn


class GRUPooledClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        h, _ = self.gru(x)
        if frame_mask is None:
            frame_mask = torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)
        m = frame_mask.unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1.0)
        pooled = (h * m).sum(dim=1) / denom
        return self.head(pooled).squeeze(-1)
