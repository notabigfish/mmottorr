from __future__ import annotations

import torch
from .pose_loss import masked_coord_mse, masked_ligand_rmsd
from .confidence_loss import confidence_bce_loss
from .motor_pair_loss import masked_pair_motor_loss
from motordock.geometry.se3 import se3_geodesic_loss


def motordock_se3_loss(
    outputs: dict,
    batch: dict,
    coord_weight: float = 1.0,
    ligand_se3_weight: float = 0.1,
    pair_motor_weight: float = 0.05,
    confidence_weight: float = 0.0,
) -> dict:
    coord = masked_coord_mse(outputs["ligand_coords_pred"], batch["ligand_coords_true"], batch["ligand_mask"])
    lig_se3 = se3_geodesic_loss(outputs["delta_T_pred"], batch["T_target"])
    pair_motor = masked_pair_motor_loss(
        outputs["pair_delta_T_pred"],
        batch["pair_T_target_residual"],
        batch["pair_mask"],
        batch.get("pair_was_perturbed", None),
    )
    rmsd = masked_ligand_rmsd(outputs["ligand_coords_pred"], batch["ligand_coords_true"], batch["ligand_mask"])
    conf = confidence_bce_loss(outputs["confidence_logit"], rmsd) if confidence_weight > 0 else coord * 0.0

    total = coord_weight * coord + ligand_se3_weight * lig_se3 + pair_motor_weight * pair_motor + confidence_weight * conf
    return {
        "total": total,
        "coord_loss": coord,
        "ligand_se3_loss": lig_se3,
        "pair_motor_loss": pair_motor,
        "confidence_loss": conf,
        "rmsd": rmsd,
    }
