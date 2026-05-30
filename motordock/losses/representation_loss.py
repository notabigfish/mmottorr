from __future__ import annotations

import torch
from motordock.geometry.se3 import se3_geodesic_loss
from motordock.geometry.pga_motor import (
    se3_to_motor,
    sandwich_points,
    full_to_even_motor,
    unit_motor_regularization,
)


def get_first_available(mapping: dict, names: list[str]):
    for n in names:
        if n in mapping and mapping[n] is not None:
            return mapping[n]
    return None


def _canonical_points(device, dtype):
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )


def pga_motor_loss(
    M_pred: torch.Tensor,
    M_target: torch.Tensor,
    T_pred: torch.Tensor | None = None,
    T_target: torch.Tensor | None = None,
    canonical_points: torch.Tensor | None = None,
    lambda_geodesic: float = 1.0,
    lambda_action: float = 0.1,
    lambda_coeff: float = 0.01,
    lambda_unit: float = 0.001,
    sigma_R: float = 0.2617993877991494,
    sigma_t: float = 2.0,
) -> torch.Tensor:
    if M_target is None and T_target is not None:
        M_target = se3_to_motor(T_target)

    if M_target is None:
        return M_pred.sum() * 0.0

    losses = M_pred.sum() * 0.0

    if T_pred is not None and T_target is not None:
        losses = losses + lambda_geodesic * se3_geodesic_loss(T_pred, T_target, sigma_R=sigma_R, sigma_t=sigma_t, reduction="mean")

    pts = canonical_points
    if pts is None:
        pts = _canonical_points(M_pred.device, M_pred.dtype)

    Mpf = M_pred.reshape(-1, 16)
    Mtf = M_target.reshape(-1, 16)
    P = pts.view(1, pts.shape[0], 3).expand(Mpf.shape[0], -1, -1)
    Pp = sandwich_points(Mpf, P)
    Pt = sandwich_points(Mtf, P)
    action = ((Pp - Pt) ** 2).sum(dim=-1).mean()

    dot = (M_pred * M_target).sum(dim=-1, keepdim=True)
    M_target_aligned = torch.where(dot < 0, -M_target, M_target)
    coeff = torch.mean((full_to_even_motor(M_pred) - full_to_even_motor(M_target_aligned)) ** 2)

    unit = unit_motor_regularization(M_pred)

    losses = losses + lambda_action * action + lambda_coeff * coeff + lambda_unit * unit
    return losses


def masked_representation_motor_loss(
    outputs: dict,
    batch: dict,
    representation: str,
    pair_mask: torch.Tensor,
    pair_was_perturbed: torch.Tensor | None = None,
    sigma_R: float = 0.2617993877991494,
    sigma_t: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    if representation in {"centroid_bias", "no_pair_context"}:
        return outputs["pair_delta_T_pred"].sum() * 0.0

    m = pair_mask
    if pair_was_perturbed is not None:
        m = m & pair_was_perturbed
    if "pair_valid" in batch:
        m = m & batch["pair_valid"]

    B, C = m.shape
    mf = m.view(B * C)
    if not torch.any(mf):
        anchor = outputs.get("pair_delta_T_pred", None)
        if anchor is None:
            anchor = outputs.get("delta_T_pred")
        return anchor.sum() * 0.0

    if representation in {"pga_feature", "pga_sandwich", "motordock_pga"}:
        M_pred_all = get_first_available(outputs, ["pga_motor"])
        T_pred_all = get_first_available(outputs, ["pair_T_corrected", "pair_delta_T_pred"])
        T_tgt_all = get_first_available(batch, ["pair_T_target_residual", "pair_T_target", "T_target"])
        M_tgt_all = get_first_available(batch, ["target_pga_motor"])

        if M_pred_all is None:
            return outputs["pair_delta_T_pred"].sum() * 0.0

        M_pred = M_pred_all.view(B * C, 16)[mf]

        M_tgt = None
        if M_tgt_all is not None:
            M_tgt = M_tgt_all.view(B * C, 16)[mf]

        T_pred = None
        if T_pred_all is not None and T_pred_all.shape[-2:] == (4, 4):
            T_pred = T_pred_all.view(B * C, 4, 4)[mf]

        T_tgt = None
        if T_tgt_all is not None and T_tgt_all.shape[-2:] == (4, 4):
            if T_tgt_all.dim() == 3:  # [B,4,4]
                T_tgt_all = T_tgt_all[:, None, :, :].expand(B, C, 4, 4)
            T_tgt = T_tgt_all.view(B * C, 4, 4)[mf]

        return pga_motor_loss(
            M_pred=M_pred,
            M_target=M_tgt,
            T_pred=T_pred,
            T_target=T_tgt,
            lambda_geodesic=float(batch.get("lambda_pga_geodesic", 1.0)) if isinstance(batch, dict) else 1.0,
            lambda_action=float(batch.get("lambda_pga_action", 0.1)) if isinstance(batch, dict) else 0.1,
            lambda_coeff=float(batch.get("lambda_pga_coeff", 0.01)) if isinstance(batch, dict) else 0.01,
            lambda_unit=float(batch.get("lambda_pga_unit", 0.001)) if isinstance(batch, dict) else 0.001,
            sigma_R=sigma_R,
            sigma_t=sigma_t,
        )

    pred = outputs["pair_delta_T_pred"].view(B * C, 4, 4)[mf]
    tgt = batch["pair_T_target_residual"].view(B * C, 4, 4)[mf]
    return se3_geodesic_loss(pred, tgt, sigma_R=sigma_R, sigma_t=sigma_t, reduction=reduction)
