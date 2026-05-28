from __future__ import annotations

from dataclasses import dataclass
import torch

from .se3 import make_transform, relative_transform


@dataclass
class FrameResult:
    R: torch.Tensor | None
    t: torch.Tensor | None
    stable: bool
    reason: str
    eigvals: torch.Tensor
    num_points: int
    det: float | None
    orthogonality_error: float | None


def rotation_orthogonality_error(R: torch.Tensor) -> torch.Tensor:
    """Return ||R^T R - I||_F."""
    I = torch.eye(3, dtype=R.dtype, device=R.device)
    return torch.linalg.norm(R.transpose(-1, -2) @ R - I, ord="fro")


def rotation_det(R: torch.Tensor) -> torch.Tensor:
    """Return det(R)."""
    return torch.det(R)


def weighted_pca_frame(
    coords: torch.Tensor,
    pocket_center: torch.Tensor,
    min_points: int = 8,
    eig_ratio_threshold: float = 0.15,
    eps: float = 1e-8,
) -> FrameResult:
    """Build deterministic local frame from C-alpha coordinates."""
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f"coords must be [N,3], got {tuple(coords.shape)}")
    if pocket_center.shape != (3,):
        raise ValueError(f"pocket_center must be [3], got {tuple(pocket_center.shape)}")

    n = int(coords.shape[0])
    eigvals_zero = torch.zeros(3, dtype=coords.dtype, device=coords.device)
    if n < min_points:
        return FrameResult(None, None, False, "too_few_points", eigvals_zero, n, None, None)

    d = torch.linalg.norm(coords - pocket_center.unsqueeze(0), dim=-1)
    w = 1.0 / (d + eps)
    w = w / (w.sum() + eps)

    t = torch.sum(coords * w.unsqueeze(-1), dim=0)
    c = coords - t.unsqueeze(0)
    cov = (c * w.unsqueeze(-1)).T @ c

    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    if eigvals[0] <= eps:
        return FrameResult(None, t, False, "zero_primary_eigenvalue", eigvals, n, None, None)

    x = eigvecs[:, 0]
    y = eigvecs[:, 1]
    y = y - x * torch.dot(x, y)
    yn = torch.linalg.norm(y)
    if yn <= eps:
        return FrameResult(None, t, False, "degenerate_second_axis", eigvals, n, None, None)
    y = y / yn

    z = torch.cross(x, y, dim=0)
    zn = torch.linalg.norm(z)
    if zn <= eps:
        return FrameResult(None, t, False, "degenerate_cross_axis", eigvals, n, None, None)
    z = z / zn

    if torch.dot(x, pocket_center - t) < 0:
        x = -x

    z = torch.cross(x, y, dim=0)
    z = z / (torch.linalg.norm(z) + eps)
    y = torch.cross(z, x, dim=0)
    y = y / (torch.linalg.norm(y) + eps)

    R = torch.stack([x, y, z], dim=-1)
    det = rotation_det(R)
    if det < 0:
        z = -z
        y = torch.cross(z, x, dim=0)
        y = y / (torch.linalg.norm(y) + eps)
        R = torch.stack([x, y, z], dim=-1)
        det = rotation_det(R)

    ratio = eigvals[1] / (eigvals[0] + eps)
    stable = True
    reason = "ok"
    if ratio < eig_ratio_threshold:
        stable = False
        reason = f"eig_ratio_below_threshold:{float(ratio):.6f}"

    ortho = rotation_orthogonality_error(R)
    return FrameResult(
        R=R,
        t=t,
        stable=stable,
        reason=reason,
        eigvals=eigvals,
        num_points=n,
        det=float(det),
        orthogonality_error=float(ortho),
    )


def make_frame_matrix(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Return 4x4 SE(3) frame matrix."""
    return make_transform(R, t)


def frame_from_saved_dict(frame_dict: dict) -> torch.Tensor | None:
    """Convert saved frame dict to 4x4 transform if stable and has R/t."""
    if not bool(frame_dict.get("stable", False)):
        return None
    R = frame_dict.get("R", None)
    t = frame_dict.get("t", None)
    if R is None or t is None:
        return None
    if not torch.is_tensor(R):
        R = torch.tensor(R)
    if not torch.is_tensor(t):
        t = torch.tensor(t)
    return make_frame_matrix(R, t)


def compute_pair_transform_from_frames(frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
    """Return frame_a^{-1} @ frame_b."""
    return relative_transform(frame_a, frame_b)
