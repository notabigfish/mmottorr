from __future__ import annotations

import torch
from motordock.geometry.se3 import split_transform, make_transform, se3_geodesic_loss, project_to_so3


def transform_to_matrix_features(T: torch.Tensor, mode: str = "3x4") -> torch.Tensor:
    if mode == "3x4":
        return T[..., :3, :4].reshape(*T.shape[:-2], 12)
    if mode == "4x4":
        return T.reshape(*T.shape[:-2], 16)
    if mode == "rot6d_trans":
        R, t = split_transform(T)
        return torch.cat([R[..., :, :2].reshape(*R.shape[:-2], 6), t], dim=-1)
    raise ValueError(mode)


def _rot6d_to_R(v: torch.Tensor) -> torch.Tensor:
    a1 = v[..., :3]
    a2 = v[..., 3:6]
    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_features_to_transform(features: torch.Tensor, mode: str = "3x4") -> torch.Tensor:
    if mode == "3x4":
        m = features.view(*features.shape[:-1], 3, 4)
        R = project_to_so3(m[..., :3])
        t = m[..., :3, 3]
        return make_transform(R, t)
    if mode == "4x4":
        m = features.view(*features.shape[:-1], 4, 4)
        R = project_to_so3(m[..., :3, :3])
        t = m[..., :3, 3]
        return make_transform(R, t)
    if mode == "rot6d_trans":
        R = _rot6d_to_R(features[..., :6])
        t = features[..., 6:9]
        return make_transform(project_to_so3(R), t)
    raise ValueError(mode)


def matrix_representation_loss(pred_features: torch.Tensor, target_features: torch.Tensor, mode: str = "3x4", sigma_R: float = 0.2617993877991494, sigma_t: float = 2.0, reduction: str = "mean") -> torch.Tensor:
    Tp = matrix_features_to_transform(pred_features, mode=mode)
    Tt = matrix_features_to_transform(target_features, mode=mode)
    return se3_geodesic_loss(Tp, Tt, sigma_R=sigma_R, sigma_t=sigma_t, reduction=reduction)
