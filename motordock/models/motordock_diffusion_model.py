from __future__ import annotations

import torch
import torch.nn as nn

from .common_layers import MLP, masked_mean
from .pair_encoder import PairEncoder, PairAttentionPool
from .motor_adapter_se3 import MotorAdapterSE3
from .pga_adapter import PGAFeatureAdapter, PGASandwichAdapter


class MotorDockDiffusionModel(nn.Module):
    def __init__(
        self,
        protein_feat_dim: int,
        ligand_feat_dim: int,
        pair_feat_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        sigma_emb_dim: int = 64,
        use_pair_attention: bool = True,
        adapter_type: str = "se3",  # se3, pga_feature, pga_sandwich, none
        disable_pair_context: bool = False,
        use_motor_auxiliary: bool = True,
        max_pair_rotation_scale: float = 0.5,
        max_pair_translation_scale: float = 5.0,
    ):
        super().__init__()
        self.use_pair_attention = use_pair_attention
        self.adapter_type = adapter_type
        self.disable_pair_context = disable_pair_context
        self.use_motor_auxiliary = use_motor_auxiliary

        self.protein_enc = MLP(protein_feat_dim + 4, hidden_dim, hidden_dim, num_layers=max(2, num_layers // 2), dropout=dropout)
        self.ligand_enc = MLP(ligand_feat_dim + 4, hidden_dim, hidden_dim, num_layers=max(2, num_layers // 2), dropout=dropout)

        self.sigma_mlp = MLP(4, sigma_emb_dim, sigma_emb_dim, num_layers=2, dropout=dropout)
        self.t_to_hidden = nn.Linear(sigma_emb_dim, hidden_dim)

        if adapter_type != "none":
            self.pair_enc = PairEncoder(pair_feat_dim, hidden_dim, dropout)
            self.pair_pool = PairAttentionPool(hidden_dim)
            self.motor = MotorAdapterSE3(
                pair_hidden_dim=hidden_dim,
                joint_hidden_dim=hidden_dim,
                max_rotation_scale=max_pair_rotation_scale,
                max_translation_scale=max_pair_translation_scale,
            )
            if adapter_type == "pga_feature":
                self.pga_adapter = PGAFeatureAdapter(in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=hidden_dim)
                self.pga_proj = nn.Linear(hidden_dim, hidden_dim)
            elif adapter_type == "pga_sandwich":
                self.pga_adapter = PGASandwichAdapter(in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=hidden_dim)
                self.pga_proj = nn.Linear(hidden_dim, hidden_dim)
            else:
                self.pga_adapter = None
                self.pga_proj = None
        else:
            self.pair_enc = None
            self.pair_pool = None
            self.motor = None
            self.pga_adapter = None
            self.pga_proj = None

        self.motor_summary_mlp = MLP(8, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)

        if disable_pair_context or adapter_type == "none":
            fuse_in = hidden_dim * 2 + sigma_emb_dim
        else:
            fuse_in = hidden_dim * 4 + sigma_emb_dim
        self.fuse = MLP(fuse_in, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)

        self.tr_head = nn.Linear(hidden_dim, 3)
        self.rot_head = nn.Linear(hidden_dim, 3)
        self.conf_head = nn.Linear(hidden_dim, 1)
        self.contact_head = nn.Linear(hidden_dim, 1)

        self.tor_mlp = MLP(hidden_dim * 4 + hidden_dim + sigma_emb_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.tor_head = nn.Linear(hidden_dim, 1)

    def forward(self, batch: dict) -> dict:
        pc = batch["pocket_center"][:, None, :]

        p_rel = batch["protein_ca"] - pc
        p_dist = torch.linalg.norm(p_rel, dim=-1, keepdim=True)
        p_h = self.protein_enc(torch.cat([batch["protein_feat"], p_rel, p_dist], dim=-1))

        lig_coords = batch.get("ligand_coords_t", batch["ligand_coords_start"])
        l_rel = lig_coords - pc
        l_dist = torch.linalg.norm(l_rel, dim=-1, keepdim=True)
        l_h = self.ligand_enc(torch.cat([batch["ligand_atom_feat"], l_rel, l_dist], dim=-1))

        pocket_res = (batch["pocket_mask"] > 0) & batch["protein_mask"]
        has_pocket = pocket_res.any(dim=1)
        p_ctx = torch.where(
            has_pocket[:, None],
            masked_mean(p_h, pocket_res, dim=1),
            masked_mean(p_h, batch["protein_mask"], dim=1),
        )
        l_ctx = masked_mean(l_h, batch["ligand_mask"], dim=1)

        sigma_tr = batch["sigma_tr"].view(-1, 1)
        sigma_rot = batch["sigma_rot"].view(-1, 1)
        sigma_in = torch.cat(
            [
                sigma_tr,
                sigma_rot,
                torch.log(sigma_tr.clamp_min(1e-8)),
                torch.log(sigma_rot.clamp_min(1e-8)),
            ],
            dim=-1,
        )
        t_emb = self.sigma_mlp(sigma_in)

        B = p_ctx.shape[0]
        pair_h = None
        pair_attn = None
        pair_xi_pred = None
        pair_delta_T_pred = None
        pair_T_corrected = None
        xi_weighted = torch.zeros(B, 6, device=p_ctx.device, dtype=p_ctx.dtype)
        omega_norm = torch.zeros(B, 1, device=p_ctx.device, dtype=p_ctx.dtype)
        trans_norm = torch.zeros(B, 1, device=p_ctx.device, dtype=p_ctx.dtype)
        motor_summary = torch.zeros(B, p_ctx.shape[-1], device=p_ctx.device, dtype=p_ctx.dtype)
        pair_ctx = torch.zeros_like(p_ctx)

        if self.adapter_type != "none" and not self.disable_pair_context:
            pair_features = batch["pair_features"]
            pair_mask = batch["pair_mask"]
            pair_h = self.pair_enc(pair_features, pair_mask)

            motor_out = self.motor(pair_h, l_ctx, p_ctx, batch["pair_T_input"], pair_mask)
            pair_xi_pred = motor_out["pair_xi_pred"]
            pair_delta_T_pred = motor_out["pair_delta_T_pred"]
            pair_T_corrected = motor_out["pair_T_corrected"]

            if self.adapter_type == "pga_feature":
                pga_out = self.pga_adapter({"pair_T_corrected": pair_T_corrected, "pair_h": pair_h})
                pair_h = pair_h + self.pga_proj(pga_out["pga_context"])
            elif self.adapter_type == "pga_sandwich":
                pga_out = self.pga_adapter(
                    {
                        "pair_T_corrected": pair_T_corrected,
                        "pair_h": pair_h,
                        "domain_points_b": batch.get("domain_points_b", None),
                    }
                )
                pair_h = pair_h + self.pga_proj(pga_out["pga_context"])

            if self.use_pair_attention:
                query = l_ctx + p_ctx + self.t_to_hidden(t_emb)
                pair_ctx, pair_attn = self.pair_pool(pair_h, query, pair_mask)
            else:
                den = pair_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
                pair_attn = pair_mask.float() / den
                pair_ctx = (pair_h * pair_attn.unsqueeze(-1)).sum(dim=1)

            xi_weighted = (pair_xi_pred * pair_attn.unsqueeze(-1)).sum(dim=1)
            omega_norm = torch.linalg.norm(xi_weighted[:, :3], dim=-1, keepdim=True)
            trans_norm = torch.linalg.norm(xi_weighted[:, 3:], dim=-1, keepdim=True)
            motor_summary = self.motor_summary_mlp(torch.cat([xi_weighted, omega_norm, trans_norm], dim=-1))

        if self.disable_pair_context or self.adapter_type == "none":
            joint_in = torch.cat([p_ctx, l_ctx, t_emb], dim=-1)
        else:
            joint_in = torch.cat([p_ctx, l_ctx, pair_ctx, motor_summary, t_emb], dim=-1)

        joint = self.fuse(joint_in)

        tr_score = self.tr_head(joint)
        rot_score = self.rot_head(joint)
        conf = self.conf_head(joint).squeeze(-1)
        contact_logit = self.contact_head(joint).squeeze(-1)

        # torsion head
        j_idx = batch.get("torsion_bond_atom_j", None)
        k_idx = batch.get("torsion_bond_atom_k", None)
        valid = batch.get("torsion_valid_mask", None)
        if j_idx is not None and k_idx is not None and valid is not None:
            B2, M = j_idx.shape
            if M > 0:
                g_j = j_idx.unsqueeze(-1).expand(-1, -1, l_h.shape[-1])
                g_k = k_idx.unsqueeze(-1).expand(-1, -1, l_h.shape[-1])
                h_j = torch.gather(l_h, 1, g_j)
                h_k = torch.gather(l_h, 1, g_k)
                pair_rep = pair_ctx.unsqueeze(1).expand(-1, M, -1)
                t_rep = t_emb.unsqueeze(1).expand(-1, M, -1)
                h_tor = torch.cat([h_j, h_k, h_j * h_k, h_j - h_k, pair_rep, t_rep], dim=-1)
                tor_score = self.tor_head(self.tor_mlp(h_tor)).squeeze(-1)
                tor_score = torch.where(valid, tor_score, torch.zeros_like(tor_score))
            else:
                tor_score = torch.zeros((B2, 0), device=joint.device, dtype=joint.dtype)
        else:
            tor_score = torch.zeros((B, 0), device=joint.device, dtype=joint.dtype)

        return {
            "tr_score_pred": tr_score,
            "rot_score_pred": rot_score,
            "tor_score_pred": tor_score,
            "confidence_logit": conf,
            "contact_logit": contact_logit,
            "pair_h": pair_h,
            "pair_attention": pair_attn,
            "pair_xi_pred": pair_xi_pred,
            "pair_delta_T_pred": pair_delta_T_pred,
            "pair_T_corrected": pair_T_corrected,
            "selected_pair_xi": xi_weighted,
            "motor_omega_norm": omega_norm.squeeze(-1),
            "motor_translation_norm": trans_norm.squeeze(-1),
        }
