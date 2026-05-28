from __future__ import annotations

import torch
from motordock.geometry.se3 import se3_geodesic_loss


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
