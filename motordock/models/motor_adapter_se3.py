from __future__ import annotations

import torch
import torch.nn as nn
from motordock.geometry.se3 import se3_exp_map


class MotorAdapterSE3(nn.Module):
    def __init__(self, pair_hidden_dim: int = 256, joint_hidden_dim: int = 256, max_rotation_scale: float = 0.5, max_translation_scale: float = 5.0):
        super().__init__()
        self.max_rotation_scale = max_rotation_scale
        self.max_translation_scale = max_translation_scale
        self.mlp = nn.Sequential(
            nn.Linear(pair_hidden_dim + joint_hidden_dim * 2, joint_hidden_dim),
            nn.ReLU(),
            nn.Linear(joint_hidden_dim, 6),
        )

    def forward(self, pair_h: torch.Tensor, ligand_context: torch.Tensor, protein_context: torch.Tensor, pair_T_input: torch.Tensor, pair_mask: torch.Tensor) -> dict:
        B, C, H = pair_h.shape
        pc = protein_context.unsqueeze(1).expand(B, C, -1)
        lc = ligand_context.unsqueeze(1).expand(B, C, -1)
        raw = self.mlp(torch.cat([pair_h, lc, pc], dim=-1))

        omega = torch.tanh(raw[..., :3]) * self.max_rotation_scale
        v = torch.tanh(raw[..., 3:]) * self.max_translation_scale
        xi = torch.cat([omega, v], dim=-1)

        delta = se3_exp_map(xi.view(B * C, 6)).view(B, C, 4, 4)
        corrected = delta @ pair_T_input

        I = torch.eye(4, dtype=pair_h.dtype, device=pair_h.device)
        invalid = ~pair_mask
        xi = xi.masked_fill(invalid.unsqueeze(-1), 0.0)
        delta = torch.where(invalid.unsqueeze(-1).unsqueeze(-1), I.view(1, 1, 4, 4), delta)
        corrected = torch.where(invalid.unsqueeze(-1).unsqueeze(-1), I.view(1, 1, 4, 4), corrected)

        return {
            "pair_xi_pred": xi,
            "pair_delta_T_pred": delta,
            "pair_T_corrected": corrected,
        }
