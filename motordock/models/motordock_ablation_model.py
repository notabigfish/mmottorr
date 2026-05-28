from __future__ import annotations

import torch
import torch.nn as nn
from motordock.data.pose_noise import apply_transform_to_points
from motordock.geometry.se3 import se3_exp_map
from .common_layers import MLP, masked_mean
from .pair_encoder import PairEncoder, PairAttentionPool
from .representation_registry import get_representation_spec
from .representation_adapters import (
    SE3LogAdapter, QuaternionTranslationAdapter, DualQuaternionAdapter, MatrixAdapter,
    CentroidBiasAdapter, RandomMotorAdapter, PGAFeatureAdapter,
)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MotorDockAblationModel(nn.Module):
    def __init__(self, protein_feat_dim, ligand_feat_dim, pair_feat_dim, representation="se3_log", matrix_mode="3x4", hidden_dim=256, num_layers=4, dropout=0.1, use_pair_attention=True, disable_pair_context=False, parameter_budget_mode="matched", max_rotation_scale=0.5, max_translation_scale=5.0):
        super().__init__()
        self.representation_name = representation
        self.disable_pair_context = disable_pair_context
        self.use_pair_attention = use_pair_attention
        self.parameter_budget_mode = parameter_budget_mode

        self.protein_enc = MLP(protein_feat_dim + 4, hidden_dim, hidden_dim, num_layers=max(2, num_layers // 2), dropout=dropout)
        self.ligand_enc = MLP(ligand_feat_dim + 4, hidden_dim, hidden_dim, num_layers=max(2, num_layers // 2), dropout=dropout)
        self.pair_enc = PairEncoder(pair_feat_dim, hidden_dim, dropout)
        self.pair_pool = PairAttentionPool(hidden_dim)

        if representation in {"se3_log", "shuffled_pairs", "no_pair_context"}:
            self.adapter = SE3LogAdapter(pair_hidden_dim=hidden_dim, joint_hidden_dim=hidden_dim, max_rotation_scale=max_rotation_scale, max_translation_scale=max_translation_scale)
        elif representation == "quaternion_translation":
            self.adapter = QuaternionTranslationAdapter(pair_hidden_dim=hidden_dim, joint_hidden_dim=hidden_dim, max_rotation_scale=max_rotation_scale, max_translation_scale=max_translation_scale)
        elif representation == "dual_quaternion":
            self.adapter = DualQuaternionAdapter(pair_hidden_dim=hidden_dim, joint_hidden_dim=hidden_dim, max_rotation_scale=max_rotation_scale, max_translation_scale=max_translation_scale)
        elif representation == "matrix":
            self.adapter = MatrixAdapter(matrix_mode=matrix_mode, pair_hidden_dim=hidden_dim, joint_hidden_dim=hidden_dim, max_rotation_scale=max_rotation_scale, max_translation_scale=max_translation_scale)
        elif representation == "centroid_bias":
            self.adapter = CentroidBiasAdapter()
        elif representation == "random_motor":
            self.adapter = RandomMotorAdapter(pair_hidden_dim=hidden_dim, joint_hidden_dim=hidden_dim, max_rotation_scale=max_rotation_scale, max_translation_scale=max_translation_scale)
        elif representation == "pga_feature":
            self.adapter = PGAFeatureAdapter()
        else:
            raise ValueError(representation)

        fuse_in = hidden_dim * (2 if disable_pair_context else 3)
        self.fuse = MLP(fuse_in, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.xi_head = nn.Linear(hidden_dim, 6)
        self.conf_head = nn.Linear(hidden_dim, 1)

    def forward(self, batch: dict) -> dict:
        pc = batch["pocket_center"][:, None, :]

        p_rel = batch["protein_ca"] - pc
        p_dist = torch.linalg.norm(p_rel, dim=-1, keepdim=True)
        p_h = self.protein_enc(torch.cat([batch["protein_feat"], p_rel, p_dist], dim=-1))

        l_rel = batch["ligand_coords_start"] - pc
        l_dist = torch.linalg.norm(l_rel, dim=-1, keepdim=True)
        l_h = self.ligand_enc(torch.cat([batch["ligand_atom_feat"], l_rel, l_dist], dim=-1))

        p_mask = batch["protein_mask"]
        pocket_res = (batch["pocket_mask"] > 0) & p_mask
        has_pocket = pocket_res.any(dim=1)
        p_ctx = torch.where(has_pocket[:, None], masked_mean(p_h, pocket_res, dim=1), masked_mean(p_h, p_mask, dim=1))
        l_ctx = masked_mean(l_h, batch["ligand_mask"], dim=1)

        pair_h = self.pair_enc(batch["pair_features"], batch["pair_mask"])
        ad = self.adapter(pair_h, l_ctx, p_ctx, batch["pair_T_input"], batch["pair_mask"])

        if self.use_pair_attention:
            pair_ctx, pair_attn = self.pair_pool(pair_h, p_ctx + l_ctx, batch["pair_mask"])
        else:
            den = batch["pair_mask"].float().sum(dim=1, keepdim=True).clamp_min(1.0)
            pair_attn = batch["pair_mask"].float() / den
            pair_ctx = (pair_h * pair_attn.unsqueeze(-1)).sum(dim=1)

        if self.disable_pair_context:
            joint = self.fuse(torch.cat([p_ctx, l_ctx], dim=-1))
        else:
            joint = self.fuse(torch.cat([p_ctx, l_ctx, pair_ctx], dim=-1))

        xi = self.xi_head(joint)
        delta = se3_exp_map(xi)
        lig_pred = apply_transform_to_points(delta, batch["ligand_coords_start"])
        conf = self.conf_head(joint).squeeze(-1)

        return {
            "xi_pred": xi,
            "delta_T_pred": delta,
            "ligand_coords_pred": lig_pred,
            "confidence_logit": conf,
            "pair_h": pair_h,
            "pair_attention": pair_attn,
            "pair_rep_pred": ad["pair_rep_pred"],
            "pair_xi_pred": ad["pair_xi_pred"],
            "pair_delta_T_pred": ad["pair_delta_T_pred"],
            "pair_T_corrected": ad["pair_T_corrected"],
            "representation_name": self.representation_name,
        }
