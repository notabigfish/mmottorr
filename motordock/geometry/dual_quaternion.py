from __future__ import annotations

import torch
from motordock.geometry.quaternion import matrix_to_quaternion, quaternion_to_matrix, normalize_quaternion, standardize_quaternion_sign
from motordock.geometry.se3 import split_transform, make_transform, se3_geodesic_loss


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1,x1,y1,z1 = q1.unbind(-1)
    w2,x2,y2,z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def normalize_dual_quaternion(dq: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    qr = normalize_quaternion(dq[..., :4], eps)
    qd = dq[..., 4:]
    return standardize_dual_quaternion_sign(torch.cat([qr, qd], dim=-1))


def standardize_dual_quaternion_sign(dq: torch.Tensor) -> torch.Tensor:
    s = torch.where(dq[..., :1] < 0, -1.0, 1.0)
    return dq * s


def transform_to_dual_quaternion(T: torch.Tensor) -> torch.Tensor:
    R, t = split_transform(T)
    qr = matrix_to_quaternion(R)
    tq = torch.cat([torch.zeros_like(t[..., :1]), t], dim=-1)
    qd = 0.5 * quaternion_multiply(tq, qr)
    return normalize_dual_quaternion(torch.cat([qr, qd], dim=-1))


def dual_quaternion_to_transform(dq: torch.Tensor) -> torch.Tensor:
    dq = normalize_dual_quaternion(dq)
    qr = dq[..., :4]
    qd = dq[..., 4:]
    R = quaternion_to_matrix(qr)
    tq = 2.0 * quaternion_multiply(qd, quaternion_conjugate(qr))
    t = tq[..., 1:]
    return make_transform(R, t)


def dual_quaternion_multiply(dq1: torch.Tensor, dq2: torch.Tensor) -> torch.Tensor:
    qr1, qd1 = dq1[..., :4], dq1[..., 4:]
    qr2, qd2 = dq2[..., :4], dq2[..., 4:]
    qr = quaternion_multiply(qr1, qr2)
    qd = quaternion_multiply(qr1, qd2) + quaternion_multiply(qd1, qr2)
    return normalize_dual_quaternion(torch.cat([qr, qd], dim=-1))


def dual_quaternion_inverse(dq: torch.Tensor) -> torch.Tensor:
    dq = normalize_dual_quaternion(dq)
    qr, qd = dq[..., :4], dq[..., 4:]
    qr_inv = quaternion_conjugate(qr)
    qd_inv = -quaternion_multiply(quaternion_multiply(qr_inv, qd), qr_inv)
    return normalize_dual_quaternion(torch.cat([qr_inv, qd_inv], dim=-1))


def dual_quaternion_loss(pred: torch.Tensor, target: torch.Tensor, sigma_R: float = 0.2617993877991494, sigma_t: float = 2.0, reduction: str = "mean") -> torch.Tensor:
    Tp = dual_quaternion_to_transform(pred)
    Tt = dual_quaternion_to_transform(target)
    return se3_geodesic_loss(Tp, Tt, sigma_R=sigma_R, sigma_t=sigma_t, reduction=reduction)
