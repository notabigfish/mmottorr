from __future__ import annotations

import torch
import torch.nn as nn

from .common_layers import MLP, masked_mean


class DiffusionDockingModel(nn.Module):
    def __init__(
        self,
        protein_feat_dim: int,
        ligand_feat_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        sigma_emb_dim: int = 64,
    ):
        super().__init__()
        self.protein_enc = MLP(protein_feat_dim + 4, hidden_dim, hidden_dim, num_layers=max(2, num_layers // 2), dropout=dropout)
        self.ligand_enc = MLP(ligand_feat_dim + 4, hidden_dim, hidden_dim, num_layers=max(2, num_layers // 2), dropout=dropout)
        self.sigma_mlp = MLP(4, sigma_emb_dim, sigma_emb_dim, num_layers=2, dropout=dropout)
        self.fuse = MLP(hidden_dim * 2 + sigma_emb_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.tr_head = nn.Linear(hidden_dim, 3)
        self.rot_head = nn.Linear(hidden_dim, 3)
        self.conf_head = nn.Linear(hidden_dim, 1)

    def forward(self, batch: dict) -> dict:
        pc = batch["pocket_center"][:, None, :]

        protein_ca = batch["protein_ca"]
        p_rel = protein_ca - pc
        p_dist = torch.linalg.norm(p_rel, dim=-1, keepdim=True)
        p_h = self.protein_enc(torch.cat([batch["protein_feat"], p_rel, p_dist], dim=-1))

        lig_coords = batch.get("ligand_coords_t", batch.get("ligand_coords_start"))
        l_rel = lig_coords - pc
        l_dist = torch.linalg.norm(l_rel, dim=-1, keepdim=True)
        l_h = self.ligand_enc(torch.cat([batch["ligand_atom_feat"], l_rel, l_dist], dim=-1))

        p_mask = batch["protein_mask"]
        pocket_res = (batch["pocket_mask"] > 0) & p_mask
        has_pocket = pocket_res.any(dim=1)
        p_ctx = torch.where(has_pocket[:, None], masked_mean(p_h, pocket_res, dim=1), masked_mean(p_h, p_mask, dim=1))
        l_ctx = masked_mean(l_h, batch["ligand_mask"], dim=1)

        sigma_tr = batch["sigma_tr"].view(-1, 1)
        sigma_rot = batch["sigma_rot"].view(-1, 1)
        sigma_in = torch.cat([sigma_tr, sigma_rot, torch.log(sigma_tr.clamp_min(1e-8)), torch.log(sigma_rot.clamp_min(1e-8))], dim=-1)
        t_emb = self.sigma_mlp(sigma_in)

        joint = self.fuse(torch.cat([p_ctx, l_ctx, t_emb], dim=-1))
        tr_score = self.tr_head(joint)
        rot_score = self.rot_head(joint)
        conf = self.conf_head(joint).squeeze(-1)

        return {
            "tr_score_pred": tr_score,
            "rot_score_pred": rot_score,
            "confidence_logit": conf,
        }
