from __future__ import annotations

import torch
from motordock.geometry.se3 import se3_geodesic_loss


def masked_pair_motor_loss(
    pair_delta_T_pred: torch.Tensor,
    pair_T_target_residual: torch.Tensor,
    pair_mask: torch.Tensor,
    pair_was_perturbed: torch.Tensor | None = None,
    sigma_R: float = 0.2617993877991494,
    sigma_t: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    B, C = pair_mask.shape
    pred = pair_delta_T_pred.view(B * C, 4, 4)
    tgt = pair_T_target_residual.view(B * C, 4, 4)

    mask = pair_mask.view(B * C)
    if pair_was_perturbed is not None:
        mask = mask & pair_was_perturbed.view(B * C)

    if not torch.any(mask):
        return pred.sum() * 0.0

    loss = se3_geodesic_loss(pred[mask], tgt[mask], sigma_R=sigma_R, sigma_t=sigma_t, reduction="none")
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()
