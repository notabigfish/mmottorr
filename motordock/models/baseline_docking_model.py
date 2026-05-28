from __future__ import annotations

import torch
import torch.nn as nn

from motordock.geometry.se3 import se3_exp_map
from motordock.data.pose_noise import apply_transform_to_points
from .common_layers import MLP, masked_mean


class BaselineDockingModel(nn.Module):
    def __init__(self, protein_feat_dim: int, ligand_feat_dim: int, hidden_dim: int = 256, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.protein_enc = MLP(protein_feat_dim + 4, hidden_dim, hidden_dim, num_layers=max(2, num_layers // 2), dropout=dropout)
        self.ligand_enc = MLP(ligand_feat_dim + 4, hidden_dim, hidden_dim, num_layers=max(2, num_layers // 2), dropout=dropout)
        self.fuse = MLP(hidden_dim * 2, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.xi_head = nn.Linear(hidden_dim, 6)
        self.conf_head = nn.Linear(hidden_dim, 1)

    def forward(self, batch: dict) -> dict:
        pc = batch["pocket_center"][:, None, :]

        p_rel = batch["protein_ca"] - pc
        p_dist = torch.linalg.norm(p_rel, dim=-1, keepdim=True)
        p_in = torch.cat([batch["protein_feat"], p_rel, p_dist], dim=-1)

        l_rel = batch["ligand_coords_start"] - pc
        l_dist = torch.linalg.norm(l_rel, dim=-1, keepdim=True)
        l_in = torch.cat([batch["ligand_atom_feat"], l_rel, l_dist], dim=-1)

        p_h = self.protein_enc(p_in)
        l_h = self.ligand_enc(l_in)

        p_mask = batch["protein_mask"]
        pocket_res = (batch["pocket_mask"] > 0) & p_mask
        has_pocket = pocket_res.any(dim=1)
        p_ctx_pocket = masked_mean(p_h, pocket_res, dim=1)
        p_ctx_all = masked_mean(p_h, p_mask, dim=1)
        p_ctx = torch.where(has_pocket[:, None], p_ctx_pocket, p_ctx_all)

        l_ctx = masked_mean(l_h, batch["ligand_mask"], dim=1)
        joint = self.fuse(torch.cat([p_ctx, l_ctx], dim=-1))

        xi_pred = self.xi_head(joint)
        confidence_logit = self.conf_head(joint).squeeze(-1)
        delta_T_pred = se3_exp_map(xi_pred)
        ligand_coords_pred = apply_transform_to_points(delta_T_pred, batch["ligand_coords_start"])

        return {
            "xi_pred": xi_pred,
            "delta_T_pred": delta_T_pred,
            "ligand_coords_pred": ligand_coords_pred,
            "confidence_logit": confidence_logit,
        }
