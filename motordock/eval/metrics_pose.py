from __future__ import annotations

import torch


def ligand_rmsd(pred, target, mask) -> torch.Tensor:
    m = mask.float().unsqueeze(-1)
    se = ((pred - target) ** 2).sum(dim=-1, keepdim=True) * m
    msd = se.sum(dim=1).squeeze(-1) / m.sum(dim=1).squeeze(-1).clamp_min(1.0)
    return torch.sqrt(msd.clamp_min(1e-12))


def centroid_distance(pred, target, mask) -> torch.Tensor:
    m = mask.float().unsqueeze(-1)
    p = (pred * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
    t = (target * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
    return torch.linalg.norm(p - t, dim=-1)


def success_rate(rmsd: torch.Tensor, threshold: float = 2.0) -> float:
    return float((rmsd < threshold).float().mean().item())


def topk_success(rmsd_matrix: torch.Tensor, k: int = 5, threshold: float = 2.0) -> float:
    k = min(k, rmsd_matrix.shape[1])
    best = rmsd_matrix[:, :k].min(dim=1).values
    return float((best < threshold).float().mean().item())
