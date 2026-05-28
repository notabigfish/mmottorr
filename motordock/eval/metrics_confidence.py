from __future__ import annotations

import torch


def rank_by_confidence(confidence: torch.Tensor) -> torch.Tensor:
    return torch.argsort(confidence, dim=-1, descending=True)


def top1_by_confidence(rmsd_matrix: torch.Tensor, confidence_matrix: torch.Tensor) -> torch.Tensor:
    idx = torch.argmax(confidence_matrix, dim=1)
    return rmsd_matrix[torch.arange(rmsd_matrix.shape[0]), idx]
