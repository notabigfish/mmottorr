from __future__ import annotations

import math
import torch
from motordock.geometry.se3 import se3_exp_map, inverse_transform


def apply_transform_to_points(T: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    if T.ndim == 2:
        R = T[:3, :3]
        t = T[:3, 3]
        return points @ R.transpose(0, 1) + t
    if T.ndim == 3:
        R = T[:, :3, :3]
        t = T[:, :3, 3]
        return points @ R.transpose(1, 2) + t[:, None, :]
    raise ValueError("T must be [4,4] or [B,4,4]")


def sample_random_ligand_transform(
    max_translation: float = 10.0,
    max_rotation_degrees: float = 180.0,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    axis = torch.randn(3, device=device, dtype=dtype)
    axis = axis / (axis.norm() + 1e-8)
    angle = torch.rand(1, device=device, dtype=dtype) * math.radians(max_rotation_degrees)
    omega = axis * angle
    trans_dir = torch.randn(3, device=device, dtype=dtype)
    trans_dir = trans_dir / (trans_dir.norm() + 1e-8)
    trans_mag = torch.rand(1, device=device, dtype=dtype) * max_translation
    v = trans_dir * trans_mag
    xi = torch.cat([omega, v], dim=0)
    return se3_exp_map(xi)


def randomize_ligand_pose(
    coords_true: torch.Tensor,
    pocket_center: torch.Tensor,
    max_translation: float = 10.0,
    max_rotation_degrees: float = 180.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c_true = coords_true.mean(dim=0)
    shift = pocket_center - c_true
    centered = coords_true + shift
    T_noise = sample_random_ligand_transform(
        max_translation=max_translation,
        max_rotation_degrees=max_rotation_degrees,
        device=coords_true.device,
        dtype=coords_true.dtype,
    )
    coords_start = apply_transform_to_points(T_noise, centered)
    T_target = inverse_transform(T_noise)
    # apply target should recover centered pose; then unshift to crystal frame
    # we bake shift into coordinates directly by adding after inverse.
    coords_start = coords_start - shift
    return coords_start, T_noise, T_target
