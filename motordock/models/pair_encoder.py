from __future__ import annotations

import torch
import torch.nn as nn
from .common_layers import MLP


class PairEncoder(nn.Module):
    def __init__(self, pair_feat_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.enc = MLP(pair_feat_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)

    def forward(self, pair_features: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        h = self.enc(pair_features)
        return h * pair_mask.unsqueeze(-1).float()


class PairAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, pair_h: torch.Tensor, query_h: torch.Tensor, pair_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q(query_h).unsqueeze(1)
        logits = (pair_h * q).sum(dim=-1)
        logits = logits.masked_fill(~pair_mask, -1e9)

        all_invalid = ~pair_mask.any(dim=1)
        if torch.any(all_invalid):
            logits = logits.clone()
            logits[all_invalid] = -1e9
            logits[all_invalid, 0] = 0.0

        attn = torch.softmax(logits, dim=1)
        attn = attn * pair_mask.float()
        s = attn.sum(dim=1, keepdim=True).clamp_min(1e-8)
        attn = attn / s
        if torch.any(all_invalid):
            attn = attn.clone()
            attn[all_invalid] = 0.0
            attn[all_invalid, 0] = 1.0

        ctx = (pair_h * attn.unsqueeze(-1)).sum(dim=1)
        return ctx, attn
