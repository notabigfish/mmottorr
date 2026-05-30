from __future__ import annotations

import torch
import torch.nn as nn

from motordock.geometry.pga_motor import se3_to_motor, motor_to_features, sandwich_points


class PGAFeatureAdapter(nn.Module):
    """
    Passive PGA feature baseline.
    Converts T into normalized PGA motor coefficients and encodes them.
    No sandwich action is applied.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, include_full: bool = False):
        super().__init__()
        self.include_full = include_full
        feat_dim = 16 if include_full else 8
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, pair_features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        T = pair_features.get("T_corrected", None)
        if T is None:
            T = pair_features.get("pair_T_corrected", None)
        if T is None:
            T = pair_features.get("T_initial", None)
        if T is None:
            T = pair_features.get("pair_T_initial", None)
        if T is None:
            raise KeyError("PGAFeatureAdapter requires T_corrected/T_initial")

        context = pair_features.get("pair_context", None)
        if context is None:
            context = pair_features.get("pair_h", None)
        if context is None:
            context = torch.zeros(*T.shape[:-2], 1, device=T.device, dtype=T.dtype)

        M = se3_to_motor(T)
        mfeat = motor_to_features(M, include_full=self.include_full)
        pga_context = self.mlp(torch.cat([mfeat, context], dim=-1))
        return {
            "pga_motor": M,
            "pga_motor_features": mfeat,
            "pga_context": pga_context,
        }


class PGASandwichAdapter(nn.Module):
    """
    True PGA adapter using Clifford sandwich action inside forward.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        primitive_mode: str = "canonical_points",
    ):
        super().__init__()
        self.primitive_mode = primitive_mode
        canonical = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("canonical_points", canonical)
        n_pts = canonical.shape[0]
        # motor(8) + transformed(3N) + displacement(3N) + norm summary(2)
        in_features = 8 + 6 * n_pts + 2 + in_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def _get_points(self, pair_features: dict[str, torch.Tensor], B: int, C: int, device, dtype) -> torch.Tensor:
        if "domain_points_b" in pair_features and pair_features["domain_points_b"] is not None:
            pts = pair_features["domain_points_b"]
            if pts.dim() == 3:
                pts = pts[:, None, :, :].expand(B, C, -1, -1)
            return pts.to(device=device, dtype=dtype)
        if "canonical_points" in pair_features and pair_features["canonical_points"] is not None:
            pts = pair_features["canonical_points"]
            if pts.dim() == 2:
                pts = pts.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
            elif pts.dim() == 3:
                pts = pts[:, None, :, :].expand(B, C, -1, -1)
            return pts.to(device=device, dtype=dtype)
        return self.canonical_points.to(device=device, dtype=dtype).view(1, 1, -1, 3).expand(B, C, -1, -1)

    def forward(self, pair_features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        T = pair_features.get("T_corrected", None)
        if T is None:
            T = pair_features.get("pair_T_corrected", None)
        if T is None:
            T = pair_features.get("T_initial", None)
        if T is None:
            T = pair_features.get("pair_T_initial", None)
        if T is None:
            raise KeyError("PGASandwichAdapter requires T_corrected/T_initial")

        pair_context = pair_features.get("pair_context", None)
        if pair_context is None:
            pair_context = pair_features.get("pair_h", None)
        if pair_context is None:
            pair_context = torch.zeros(*T.shape[:-2], 1, device=T.device, dtype=T.dtype)

        B, C = T.shape[:2]
        M = se3_to_motor(T)
        mfeat = motor_to_features(M, include_full=False)

        pts = self._get_points(pair_features, B, C, T.device, T.dtype)
        pts_flat = pts.reshape(B * C, pts.shape[-2], 3)
        M_flat = M.reshape(B * C, 16)

        transformed = sandwich_points(M_flat, pts_flat).reshape(B, C, pts.shape[-2], 3)
        disp = transformed - pts
        disp_norm = torch.linalg.norm(disp, dim=-1)
        action_summary = torch.stack([
            disp_norm.mean(dim=-1),
            disp_norm.max(dim=-1).values,
        ], dim=-1)

        action_features = torch.cat([
            transformed.reshape(B, C, -1),
            disp.reshape(B, C, -1),
            action_summary,
        ], dim=-1)

        fused = torch.cat([mfeat, action_features, pair_context], dim=-1)
        pga_context = self.mlp(fused)
        return {
            "pga_motor": M,
            "pga_motor_features": mfeat,
            "pga_transformed_points": transformed,
            "pga_action_features": action_features,
            "pga_context": pga_context,
        }
