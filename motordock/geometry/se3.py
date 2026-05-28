"""SE(3) utilities with PyTorch tensors.

Conventions:
- Rotation matrices are active transforms in SO(3).
- Homogeneous transforms T map points p as p' = R p + t.
- Twist vectors use xi = [omega_x, omega_y, omega_z, v_x, v_y, v_z].
"""

from __future__ import annotations

import math
import torch


def _eye3_like(x: torch.Tensor) -> torch.Tensor:
    return torch.eye(3, dtype=x.dtype, device=x.device).expand(x.shape[:-1] + (3, 3))


def skew(w: torch.Tensor) -> torch.Tensor:
    """Convert vectors (..., 3) to skew-symmetric matrices (..., 3, 3)."""
    if w.shape[-1] != 3:
        raise ValueError(f"Expected (..., 3), got {tuple(w.shape)}")
    wx, wy, wz = w.unbind(dim=-1)
    O = torch.zeros_like(wx)
    row0 = torch.stack([O, -wz, wy], dim=-1)
    row1 = torch.stack([wz, O, -wx], dim=-1)
    row2 = torch.stack([-wy, wx, O], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def unskew(W: torch.Tensor) -> torch.Tensor:
    """Convert skew-symmetric matrices (..., 3, 3) to vectors (..., 3)."""
    if W.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3), got {tuple(W.shape)}")
    return torch.stack([W[..., 2, 1], W[..., 0, 2], W[..., 1, 0]], dim=-1)


def make_transform(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Build homogeneous SE(3) matrices from R (..., 3, 3) and t (..., 3)."""
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected R (..., 3, 3), got {tuple(R.shape)}")
    if t.shape[-1] != 3:
        raise ValueError(f"Expected t (..., 3), got {tuple(t.shape)}")
    batch = torch.broadcast_shapes(R.shape[:-2], t.shape[:-1])
    Rb = R.expand(batch + (3, 3))
    tb = t.expand(batch + (3,))
    T = torch.zeros(batch + (4, 4), dtype=Rb.dtype, device=Rb.device)
    T[..., :3, :3] = Rb
    T[..., :3, 3] = tb
    T[..., 3, 3] = 1.0
    return T


def split_transform(T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split homogeneous matrices into R (..., 3, 3) and t (..., 3)."""
    if T.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (..., 4, 4), got {tuple(T.shape)}")
    return T[..., :3, :3], T[..., :3, 3]


def inverse_transform(T: torch.Tensor) -> torch.Tensor:
    """Invert SE(3) homogeneous transforms."""
    R, t = split_transform(T)
    Rt = R.transpose(-1, -2)
    tinv = -(Rt @ t.unsqueeze(-1)).squeeze(-1)
    return make_transform(Rt, tinv)


def compose_transform(T1: torch.Tensor, T2: torch.Tensor) -> torch.Tensor:
    """Compose T1 @ T2."""
    if T1.shape[-2:] != (4, 4) or T2.shape[-2:] != (4, 4):
        raise ValueError("Expected (..., 4, 4) inputs")
    return T1 @ T2


def relative_transform(Fa: torch.Tensor, Fb: torch.Tensor) -> torch.Tensor:
    """Compute Fa^{-1} Fb."""
    return compose_transform(inverse_transform(Fa), Fb)


def so3_exp_map(omega: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Exponential map from so(3) vector (..., 3) to rotation matrix (..., 3, 3)."""
    if omega.shape[-1] != 3:
        raise ValueError("omega must have shape (..., 3)")
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    theta2 = theta * theta
    theta4 = theta2 * theta2

    A_taylor = 1.0 - theta2 / 6.0 + theta4 / 120.0
    B_taylor = 0.5 - theta2 / 24.0 + theta4 / 720.0

    A = torch.where(theta > eps, torch.sin(theta) / theta, A_taylor)
    B = torch.where(theta > eps, (1.0 - torch.cos(theta)) / theta2.clamp_min(eps * eps), B_taylor)

    W = skew(omega)
    W2 = W @ W
    I = _eye3_like(omega)
    return I + A.unsqueeze(-1) * W + B.unsqueeze(-1) * W2


def so3_log_map(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Log map from rotation matrix (..., 3, 3) to rotation vector (..., 3)."""
    if R.shape[-2:] != (3, 3):
        raise ValueError("R must have shape (..., 3, 3)")
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)

    vee = unskew(R - R.transpose(-1, -2))
    scale_small = 0.5 + (theta * theta) / 12.0
    scale = torch.where(
        theta > eps,
        theta / (2.0 * sin_theta.clamp_min(eps)),
        scale_small,
    )
    omega = scale.unsqueeze(-1) * vee
    return torch.nan_to_num(omega)


def _left_jacobian_so3(omega: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    theta2 = theta * theta
    theta4 = theta2 * theta2

    B_taylor = 0.5 - theta2 / 24.0 + theta4 / 720.0
    C_taylor = (1.0 / 6.0) - theta2 / 120.0 + theta4 / 5040.0

    B = torch.where(theta > eps, (1.0 - torch.cos(theta)) / theta2.clamp_min(eps * eps), B_taylor)
    C = torch.where(theta > eps, (theta - torch.sin(theta)) / (theta2 * theta).clamp_min(eps ** 3), C_taylor)

    W = skew(omega)
    W2 = W @ W
    I = _eye3_like(omega)
    return I + B.unsqueeze(-1) * W + C.unsqueeze(-1) * W2


def _left_jacobian_inv_so3(omega: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    theta2 = theta * theta
    W = skew(omega)
    W2 = W @ W
    I = _eye3_like(omega)

    coeff_taylor = (1.0 / 12.0) + theta2 / 720.0
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    # denom = (2.0 * (1.0 - cos_t)).clamp_min(eps)
    denom = (2.0 * theta * sin_t).clamp_min(eps)
    coeff = (1.0 / theta2.clamp_min(eps * eps)) - ((1.0 + cos_t) / denom)
    coeff = torch.where(theta > eps, coeff, coeff_taylor)

    return I - 0.5 * W + coeff.unsqueeze(-1) * W2


def se3_exp_map(xi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Exponential map from xi (..., 6) to SE(3) matrix (..., 4, 4)."""
    if xi.shape[-1] != 6:
        raise ValueError("xi must have shape (..., 6)")
    omega = xi[..., :3]
    v = xi[..., 3:]
    R = so3_exp_map(omega, eps=eps)
    V = _left_jacobian_so3(omega, eps=eps)
    t = (V @ v.unsqueeze(-1)).squeeze(-1)
    return make_transform(R, t)


def se3_log_map(T: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Log map from SE(3) matrix (..., 4, 4) to xi (..., 6)."""
    R, t = split_transform(T)
    omega = so3_log_map(R, eps=eps)
    Vinv = _left_jacobian_inv_so3(omega, eps=eps)
    v = (Vinv @ t.unsqueeze(-1)).squeeze(-1)
    return torch.cat([omega, v], dim=-1)


def project_to_so3(R: torch.Tensor) -> torch.Tensor:
    """Project nearly-orthogonal matrices to SO(3) using SVD and det(+1)."""
    if R.shape[-2:] != (3, 3):
        raise ValueError("R must have shape (...,3,3)")
    U, _, Vh = torch.linalg.svd(R)
    Rproj = U @ Vh
    det = torch.det(Rproj)
    neg = det < 0
    if torch.any(neg):
        U_adj = U.clone()
        U_adj[..., :, 2] = torch.where(neg.unsqueeze(-1), -U_adj[..., :, 2], U_adj[..., :, 2])
        Rproj = U_adj @ Vh
    return Rproj


def is_valid_rotation(R: torch.Tensor, atol: float = 1e-4) -> torch.Tensor:
    """Return boolean tensor for whether R^T R ~ I and det(R)>0."""
    if R.shape[-2:] != (3, 3):
        raise ValueError("R must have shape (...,3,3)")
    I = torch.eye(3, dtype=R.dtype, device=R.device).expand(R.shape[:-2] + (3, 3))
    ortho = torch.linalg.norm(R.transpose(-1, -2) @ R - I, dim=(-2, -1))
    det = torch.det(R)
    return (ortho <= atol) & (det > 0.0)


def is_valid_transform(T: torch.Tensor, atol: float = 1e-4) -> torch.Tensor:
    """Return boolean tensor for whether T is a valid homogeneous SE(3) transform."""
    if T.shape[-2:] != (4, 4):
        raise ValueError("T must have shape (...,4,4)")
    R = T[..., :3, :3]
    bottom = T[..., 3, :]
    b_ok = torch.all(torch.isclose(bottom, torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=T.dtype, device=T.device), atol=atol), dim=-1)
    return is_valid_rotation(R, atol=atol) & b_ok


def se3_geodesic_loss(
    T_pred: torch.Tensor,
    T_true: torch.Tensor,
    sigma_R: float = 0.2617993877991494,
    sigma_t: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute xi-error weighted by rotation and translation scales."""
    T_err = compose_transform(inverse_transform(T_pred), T_true)
    xi_err = se3_log_map(T_err)
    omega = xi_err[..., :3]
    v = xi_err[..., 3:]
    rot_term = torch.sum(omega * omega, dim=-1) / (sigma_R * sigma_R)
    trans_term = torch.sum(v * v, dim=-1) / (sigma_t * sigma_t)
    loss = rot_term + trans_term

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"Unknown reduction: {reduction}")
