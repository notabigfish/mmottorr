from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(max(1, num_layers - 1)):
            layers.extend([nn.Linear(d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    m = mask.float().unsqueeze(-1)
    num = (x * m).sum(dim=dim)
    den = m.sum(dim=dim).clamp_min(1.0)
    return num / den


def masked_max(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    neg = torch.finfo(x.dtype).min
    y = x.masked_fill(~mask.unsqueeze(-1), neg)
    return y.max(dim=dim).values
