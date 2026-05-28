from __future__ import annotations

import torch
from motordock.geometry.se3 import se3_geodesic_loss


def masked_representation_motor_loss(outputs: dict, batch: dict, representation: str, pair_mask: torch.Tensor, pair_was_perturbed: torch.Tensor | None = None, sigma_R: float = 0.2617993877991494, sigma_t: float = 2.0, reduction: str = "mean") -> torch.Tensor:
    if representation in {"centroid_bias", "no_pair_context", "pga_feature"}:
        return outputs["pair_delta_T_pred"].sum() * 0.0

    m = pair_mask
    if pair_was_perturbed is not None:
        m = m & pair_was_perturbed
    if "pair_valid" in batch:
        m = m & batch["pair_valid"]

    B, C = m.shape
    mf = m.view(B * C)
    if not torch.any(mf):
        return outputs["pair_delta_T_pred"].sum() * 0.0

    pred = outputs["pair_delta_T_pred"].view(B * C, 4, 4)[mf]
    tgt = batch["pair_T_target_residual"].view(B * C, 4, 4)[mf]
    return se3_geodesic_loss(pred, tgt, sigma_R=sigma_R, sigma_t=sigma_t, reduction=reduction)
