from __future__ import annotations

import torch
import torch.nn.functional as F


def confidence_target_from_rmsd(rmsd: torch.Tensor, threshold: float = 2.0) -> torch.Tensor:
    return (rmsd < threshold).float()


def confidence_bce_loss(confidence_logit: torch.Tensor, rmsd: torch.Tensor) -> torch.Tensor:
    y = confidence_target_from_rmsd(rmsd.detach())
    return F.binary_cross_entropy_with_logits(confidence_logit, y)
