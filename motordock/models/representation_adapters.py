from __future__ import annotations

import torch
import torch.nn as nn
from motordock.geometry.se3 import se3_exp_map, se3_log_map
from motordock.geometry.representation_conversions import representation_to_transform
from motordock.geometry.quaternion import normalize_quaternion, standardize_quaternion_sign
from motordock.geometry.dual_quaternion import normalize_dual_quaternion, standardize_dual_quaternion_sign
from motordock.geometry.matrix_representation import matrix_features_to_transform
from .pga_adapter import PGAFeatureAdapter as _PGAFeatureCore, PGASandwichAdapter as _PGASandwichCore


class BaseRepresentationAdapter(nn.Module):
    def forward(self, pair_h, ligand_context, protein_context, pair_T_input, pair_mask):
        raise NotImplementedError


class _CommonAdapter(BaseRepresentationAdapter):
    def __init__(self, pair_hidden_dim=256, joint_hidden_dim=256, out_dim=6, max_rotation_scale=0.5, max_translation_scale=5.0):
        super().__init__()
        self.max_rotation_scale = max_rotation_scale
        self.max_translation_scale = max_translation_scale
        self.out_dim = out_dim
        self.mlp = nn.Sequential(nn.Linear(pair_hidden_dim + 2 * joint_hidden_dim, joint_hidden_dim), nn.ReLU(), nn.Linear(joint_hidden_dim, out_dim))

    def _ctx(self, pair_h, ligand_context, protein_context):
        B, C, _ = pair_h.shape
        lc = ligand_context.unsqueeze(1).expand(B, C, -1)
        pc = protein_context.unsqueeze(1).expand(B, C, -1)
        return torch.cat([pair_h, lc, pc], dim=-1)

    def _mask_id(self, rep, delta, corrected, pair_mask):
        I = torch.eye(4, device=delta.device, dtype=delta.dtype)
        invalid = ~pair_mask
        rep = rep.masked_fill(invalid.unsqueeze(-1), 0.0)
        delta = torch.where(invalid.unsqueeze(-1).unsqueeze(-1), I.view(1, 1, 4, 4), delta)
        corrected = torch.where(invalid.unsqueeze(-1).unsqueeze(-1), I.view(1, 1, 4, 4), corrected)
        return rep, delta, corrected


class SE3LogAdapter(_CommonAdapter):
    def __init__(self, **kw):
        super().__init__(out_dim=6, **kw)

    def forward(self, pair_h, ligand_context, protein_context, pair_T_input, pair_mask):
        raw = self.mlp(self._ctx(pair_h, ligand_context, protein_context))
        omega = torch.tanh(raw[..., :3]) * self.max_rotation_scale
        v = torch.tanh(raw[..., 3:]) * self.max_translation_scale
        rep = torch.cat([omega, v], dim=-1)
        delta = se3_exp_map(rep.view(-1, 6)).view(*rep.shape[:-1], 4, 4)
        corrected = delta @ pair_T_input
        rep, delta, corrected = self._mask_id(rep, delta, corrected, pair_mask)
        return {"pair_rep_pred": rep, "pair_xi_pred": rep, "pair_delta_T_pred": delta, "pair_T_corrected": corrected}


class QuaternionTranslationAdapter(_CommonAdapter):
    def __init__(self, **kw):
        super().__init__(out_dim=7, **kw)

    def forward(self, pair_h, ligand_context, protein_context, pair_T_input, pair_mask):
        raw = self.mlp(self._ctx(pair_h, ligand_context, protein_context))
        q = standardize_quaternion_sign(normalize_quaternion(raw[..., :4]))
        t = torch.tanh(raw[..., 4:]) * self.max_translation_scale
        rep = torch.cat([q, t], dim=-1)
        delta = representation_to_transform(rep.view(-1, 7), "quaternion_translation").view(*rep.shape[:-1], 4, 4)
        corrected = delta @ pair_T_input
        rep, delta, corrected = self._mask_id(rep, delta, corrected, pair_mask)
        xi = se3_log_map(delta.view(-1, 4, 4)).view(*rep.shape[:-1], 6)
        xi = xi.masked_fill((~pair_mask).unsqueeze(-1), 0.0)
        return {"pair_rep_pred": rep, "pair_xi_pred": xi, "pair_delta_T_pred": delta, "pair_T_corrected": corrected}


class DualQuaternionAdapter(_CommonAdapter):
    def __init__(self, **kw):
        super().__init__(out_dim=8, **kw)

    def forward(self, pair_h, ligand_context, protein_context, pair_T_input, pair_mask):
        raw = self.mlp(self._ctx(pair_h, ligand_context, protein_context))
        rep = standardize_dual_quaternion_sign(normalize_dual_quaternion(raw))
        delta = representation_to_transform(rep.view(-1, 8), "dual_quaternion").view(*rep.shape[:-1], 4, 4)
        corrected = delta @ pair_T_input
        rep, delta, corrected = self._mask_id(rep, delta, corrected, pair_mask)
        xi = se3_log_map(delta.view(-1, 4, 4)).view(*rep.shape[:-1], 6)
        xi = xi.masked_fill((~pair_mask).unsqueeze(-1), 0.0)
        return {"pair_rep_pred": rep, "pair_xi_pred": xi, "pair_delta_T_pred": delta, "pair_T_corrected": corrected}


class MatrixAdapter(_CommonAdapter):
    def __init__(self, matrix_mode="3x4", **kw):
        out_dim = 12 if matrix_mode == "3x4" else (16 if matrix_mode == "4x4" else 9)
        super().__init__(out_dim=out_dim, **kw)
        self.matrix_mode = matrix_mode

    def forward(self, pair_h, ligand_context, protein_context, pair_T_input, pair_mask):
        rep = self.mlp(self._ctx(pair_h, ligand_context, protein_context))
        delta = matrix_features_to_transform(rep.view(-1, rep.shape[-1]), mode=self.matrix_mode).view(*rep.shape[:-1], 4, 4)
        corrected = delta @ pair_T_input
        rep, delta, corrected = self._mask_id(rep, delta, corrected, pair_mask)
        xi = se3_log_map(delta.view(-1, 4, 4)).view(*rep.shape[:-1], 6)
        xi = xi.masked_fill((~pair_mask).unsqueeze(-1), 0.0)
        return {"pair_rep_pred": rep, "pair_xi_pred": xi, "pair_delta_T_pred": delta, "pair_T_corrected": corrected}


class CentroidBiasAdapter(BaseRepresentationAdapter):
    def forward(self, pair_h, ligand_context, protein_context, pair_T_input, pair_mask):
        B, C, _ = pair_h.shape
        I = torch.eye(4, device=pair_h.device, dtype=pair_h.dtype).view(1, 1, 4, 4).repeat(B, C, 1, 1)
        rep = torch.zeros(B, C, 1, device=pair_h.device, dtype=pair_h.dtype)
        return {"pair_rep_pred": rep, "pair_xi_pred": torch.zeros(B, C, 6, device=pair_h.device, dtype=pair_h.dtype), "pair_delta_T_pred": I, "pair_T_corrected": pair_T_input}


class RandomMotorAdapter(SE3LogAdapter):
    pass


class PGAFeatureAdapter(BaseRepresentationAdapter):
    def __init__(self, pair_hidden_dim=256, joint_hidden_dim=256, include_full=False, **_kw):
        super().__init__()
        self.core = _PGAFeatureCore(in_dim=pair_hidden_dim, hidden_dim=joint_hidden_dim, out_dim=pair_hidden_dim, include_full=include_full)

    def forward(self, pair_h, ligand_context, protein_context, pair_T_input, pair_mask):
        out = self.core({"pair_T_initial": pair_T_input, "pair_h": pair_h})
        B, C, _ = pair_h.shape
        I = torch.eye(4, device=pair_h.device, dtype=pair_h.dtype).view(1, 1, 4, 4).expand(B, C, 4, 4)
        xi = torch.zeros(B, C, 6, device=pair_h.device, dtype=pair_h.dtype)
        xi = xi.masked_fill((~pair_mask).unsqueeze(-1), 0.0)
        rep = out["pga_motor_features"].masked_fill((~pair_mask).unsqueeze(-1), 0.0)
        return {
            "pair_rep_pred": rep,
            "pair_xi_pred": xi,
            "pair_delta_T_pred": I,
            "pair_T_corrected": pair_T_input,
            "pga_motor": out["pga_motor"],
            "pga_motor_features": out["pga_motor_features"],
            "pga_context": out["pga_context"],
        }


class PGASandwichAdapter(BaseRepresentationAdapter):
    def __init__(self, pair_hidden_dim=256, joint_hidden_dim=256, primitive_mode="canonical_points", **_kw):
        super().__init__()
        self.se3 = SE3LogAdapter(pair_hidden_dim=pair_hidden_dim, joint_hidden_dim=joint_hidden_dim)
        self.core = _PGASandwichCore(in_dim=pair_hidden_dim, hidden_dim=joint_hidden_dim, out_dim=pair_hidden_dim, primitive_mode=primitive_mode)

    def forward(self, pair_h, ligand_context, protein_context, pair_T_input, pair_mask):
        se3_out = self.se3(pair_h, ligand_context, protein_context, pair_T_input, pair_mask)
        pga = self.core({
            "pair_T_initial": pair_T_input,
            "pair_T_corrected": se3_out["pair_T_corrected"],
            "pair_h": pair_h,
        })
        se3_out.update(pga)
        return se3_out
