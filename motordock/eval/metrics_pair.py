from __future__ import annotations

import torch
from motordock.geometry.se3 import se3_log_map


def pair_residual_errors(T_pred: torch.Tensor, T_true: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    err = se3_log_map(torch.linalg.inv(T_pred) @ T_true)
    rot = torch.linalg.norm(err[..., :3], dim=-1)
    trans = torch.linalg.norm(err[..., 3:], dim=-1)
    m = mask.float()
    r = (rot * m).sum() / m.sum().clamp_min(1.0)
    t = (trans * m).sum() / m.sum().clamp_min(1.0)
    return r, t


def attention_entropy(attn: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    p = attn.clamp_min(1e-8)
    e = -(p * p.log())
    e = e * mask.float()
    den = mask.float().sum(dim=1).clamp_min(1.0)
    return (e.sum(dim=1) / den).mean()
