from __future__ import annotations

import torch
from motordock.geometry.se3 import make_transform, split_transform, se3_geodesic_loss, project_to_so3


def normalize_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def standardize_quaternion_sign(q: torch.Tensor) -> torch.Tensor:
    s = torch.where(q[..., :1] < 0, -1.0, 1.0)
    return q * s


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = standardize_quaternion_sign(normalize_quaternion(q))
    w, x, y, z = q.unbind(dim=-1)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    R = torch.stack([
        torch.stack([ww+xx-yy-zz, 2*(xy-wz), 2*(xz+wy)], dim=-1),
        torch.stack([2*(xy+wz), ww-xx+yy-zz, 2*(yz-wx)], dim=-1),
        torch.stack([2*(xz-wy), 2*(yz+wx), ww-xx-yy+zz], dim=-1),
    ], dim=-2)
    return project_to_so3(R)


def matrix_to_quaternion(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    R = project_to_so3(R)
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    qw = torch.sqrt((1.0 + tr).clamp_min(eps)) / 2.0
    qx = (R[..., 2, 1] - R[..., 1, 2]) / (4.0 * qw.clamp_min(eps))
    qy = (R[..., 0, 2] - R[..., 2, 0]) / (4.0 * qw.clamp_min(eps))
    qz = (R[..., 1, 0] - R[..., 0, 1]) / (4.0 * qw.clamp_min(eps))
    q = torch.stack([qw, qx, qy, qz], dim=-1)
    return standardize_quaternion_sign(normalize_quaternion(q))


def transform_to_quat_trans(T: torch.Tensor) -> torch.Tensor:
    R, t = split_transform(T)
    q = matrix_to_quaternion(R)
    return torch.cat([q, t], dim=-1)


def quat_trans_to_transform(qt: torch.Tensor) -> torch.Tensor:
    q = qt[..., :4]
    t = qt[..., 4:]
    R = quaternion_to_matrix(q)
    return make_transform(R, t)


def quat_trans_geodesic_loss(pred: torch.Tensor, target: torch.Tensor, sigma_R: float = 0.2617993877991494, sigma_t: float = 2.0, reduction: str = "mean") -> torch.Tensor:
    Tp = quat_trans_to_transform(pred)
    Tt = quat_trans_to_transform(target)
    return se3_geodesic_loss(Tp, Tt, sigma_R=sigma_R, sigma_t=sigma_t, reduction=reduction)
