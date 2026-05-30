from __future__ import annotations

import torch


def _skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(dim=-1)
    o = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([o, -z, y], dim=-1),
            torch.stack([z, o, -x], dim=-1),
            torch.stack([-y, x, o], dim=-1),
        ],
        dim=-2,
    )


def _so3_exp(omega: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    I = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(omega.shape[:-1] + (3, 3))

    u = omega / theta.clamp_min(eps)
    K = _skew(u)
    K2 = K @ K
    sin_t = torch.sin(theta)[..., None]
    cos_t = torch.cos(theta)[..., None]
    R = I + sin_t * K + (1.0 - cos_t) * K2

    W = _skew(omega)
    R_small = I + W
    use_small = (theta < eps).unsqueeze(-1).expand_as(R)
    return torch.where(use_small, R_small, R)


def center_of_mass(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float().unsqueeze(-1)
    return (coords * m).sum(dim=-2) / m.sum(dim=-2).clamp_min(1.0)


def apply_rigid_update(coords: torch.Tensor, rotvec: torch.Tensor, translation: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    R = _so3_exp(rotvec)
    centered = coords - center.unsqueeze(-2)
    rotated = centered @ R.transpose(-1, -2)
    return rotated + center.unsqueeze(-2) + translation.unsqueeze(-2)


def random_rotation_vec(batch_size: int, sigma: torch.Tensor | float, device: torch.device) -> torch.Tensor:
    eps = torch.randn(batch_size, 3, device=device)
    if torch.is_tensor(sigma):
        sigma = sigma.to(device).view(batch_size, 1)
    return eps * float(sigma) if not torch.is_tensor(sigma) else eps * sigma


def random_translation(batch_size: int, sigma: torch.Tensor | float, device: torch.device) -> torch.Tensor:
    eps = torch.randn(batch_size, 3, device=device)
    if torch.is_tensor(sigma):
        sigma = sigma.to(device).view(batch_size, 1)
    return eps * float(sigma) if not torch.is_tensor(sigma) else eps * sigma


def perturb_rigid_pose(
    coords_0: torch.Tensor,
    mask: torch.Tensor,
    pocket_center: torch.Tensor,
    sigma_tr: torch.Tensor,
    sigma_rot: torch.Tensor,
):
    del pocket_center  # reserved for future center-aware init policy

    B = coords_0.shape[0]
    c0 = center_of_mass(coords_0, mask)

    eps_tr = torch.randn(B, 3, device=coords_0.device, dtype=coords_0.dtype)
    eps_rot = torch.randn(B, 3, device=coords_0.device, dtype=coords_0.dtype)

    tr_scale = sigma_tr.view(B, 1)
    rot_scale = sigma_rot.view(B, 1)

    u_t = tr_scale * eps_tr
    omega_t = rot_scale * eps_rot

    coords_t = apply_rigid_update(coords_0, omega_t, u_t, c0)

    target_tr_score = -u_t / (tr_scale.pow(2).clamp_min(1e-12))
    target_rot_score = -omega_t / (rot_scale.pow(2).clamp_min(1e-12))

    return coords_t, eps_tr, eps_rot, target_tr_score, target_rot_score


def prepare_diffusion_batch_targets(
    batch: dict,
    sigma_tr: torch.Tensor,
    sigma_rot: torch.Tensor,
) -> dict:
    coords_t, eps_tr, eps_rot, target_tr_score, target_rot_score = perturb_rigid_pose(
        batch["ligand_coords_true"],
        batch["ligand_mask"],
        batch["pocket_center"],
        sigma_tr,
        sigma_rot,
    )
    out = dict(batch)
    out["ligand_coords_t"] = coords_t
    out["eps_tr"] = eps_tr
    out["eps_rot"] = eps_rot
    out["target_tr_score"] = target_tr_score
    out["target_rot_score"] = target_rot_score
    out["sigma_tr"] = sigma_tr
    out["sigma_rot"] = sigma_rot
    # torsion extension hooks
    out.setdefault("torsion_target", None)
    out.setdefault("torsion_sigma", None)
    return out
