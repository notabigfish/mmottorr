from __future__ import annotations

import torch
from motordock.geometry.se3 import se3_geodesic_loss
from motordock.diffusion.torsion import torsion_score_loss


def masked_coord_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    m = mask.float().unsqueeze(-1)
    se = ((pred - target) ** 2) * m
    per = se.sum(dim=(-1, -2)) / m.sum(dim=(-1, -2)).clamp_min(1.0)
    if reduction == "none":
        return per
    if reduction == "sum":
        return per.sum()
    return per.mean()


def masked_ligand_rmsd(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float().unsqueeze(-1)
    se = ((pred - target) ** 2).sum(dim=-1, keepdim=True) * m
    msd = se.sum(dim=1).squeeze(-1) / m.sum(dim=1).squeeze(-1).clamp_min(1.0)
    return torch.sqrt(msd.clamp_min(1e-12))


def rigid_docking_loss(outputs: dict, batch: dict, coord_weight: float = 1.0, se3_weight: float = 0.1) -> dict:
    coord = masked_coord_mse(outputs["ligand_coords_pred"], batch["ligand_coords_true"], batch["ligand_mask"])
    se3 = se3_geodesic_loss(outputs["delta_T_pred"], batch["T_target"])
    total = coord_weight * coord + se3_weight * se3
    rmsd = masked_ligand_rmsd(outputs["ligand_coords_pred"], batch["ligand_coords_true"], batch["ligand_mask"]) 
    return {"total": total, "coord_loss": coord, "se3_loss": se3, "rmsd": rmsd}


def diffusion_rigid_loss(
    model_out: dict,
    batch_targets: dict,
    sigma_tr: torch.Tensor,
    sigma_rot: torch.Tensor,
    sigma_tor: torch.Tensor | None = None,
    lambda_tr: float = 1.0,
    lambda_rot: float = 1.0,
    lambda_tor: float = 1.0,
) -> dict:
    s_tr_hat = model_out["tr_score_pred"]
    s_rot_hat = model_out["rot_score_pred"]
    s_tr_tgt = batch_targets["target_tr_score"]
    s_rot_tgt = batch_targets["target_rot_score"]

    tr_w = sigma_tr.view(-1, 1).pow(2)
    rot_w = sigma_rot.view(-1, 1).pow(2)

    l_tr = (tr_w * (s_tr_hat - s_tr_tgt).pow(2).sum(dim=-1, keepdim=True)).mean()
    l_rot = (rot_w * (s_rot_hat - s_rot_tgt).pow(2).sum(dim=-1, keepdim=True)).mean()
    l_tor = s_tr_hat.sum() * 0.0

    if (
        "tor_score_pred" in model_out
        and "target_tor_score" in batch_targets
        and batch_targets["target_tor_score"] is not None
        and sigma_tor is not None
        and "torsion_valid_mask" in batch_targets
    ):
        l_tor = torsion_score_loss(
            model_out["tor_score_pred"],
            batch_targets["target_tor_score"],
            sigma_tor,
            batch_targets["torsion_valid_mask"],
        )

    total = lambda_tr * l_tr + lambda_rot * l_rot + lambda_tor * l_tor
    return {"total": total, "tr_loss": l_tr, "rot_loss": l_rot, "tor_loss": l_tor}
