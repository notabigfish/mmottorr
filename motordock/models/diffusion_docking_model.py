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
        self.tor_mlp = MLP(hidden_dim * 4 + sigma_emb_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.tor_head = nn.Linear(hidden_dim, 1)

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

        # torsion score head
        j_idx = batch.get("torsion_bond_atom_j", None)
        k_idx = batch.get("torsion_bond_atom_k", None)
        valid = batch.get("torsion_valid_mask", None)
        if j_idx is not None and k_idx is not None and valid is not None:
            B, M = j_idx.shape
            if M > 0:
                gather_j = j_idx.unsqueeze(-1).expand(-1, -1, l_h.shape[-1])
                gather_k = k_idx.unsqueeze(-1).expand(-1, -1, l_h.shape[-1])
                h_j = torch.gather(l_h, 1, gather_j)
                h_k = torch.gather(l_h, 1, gather_k)
                t_rep = t_emb.unsqueeze(1).expand(-1, M, -1)
                h_tor = torch.cat([h_j, h_k, h_j * h_k, h_j - h_k, t_rep], dim=-1)
                tor_score = self.tor_head(self.tor_mlp(h_tor)).squeeze(-1)
                tor_score = torch.where(valid, tor_score, torch.zeros_like(tor_score))
            else:
                tor_score = torch.zeros((B, 0), device=joint.device, dtype=joint.dtype)
        else:
            tor_score = torch.zeros((joint.shape[0], 0), device=joint.device, dtype=joint.dtype)

        return {
            "tr_score_pred": tr_score,
            "rot_score_pred": rot_score,
            "tor_score_pred": tor_score,
            "confidence_logit": conf,
        }
