from __future__ import annotations

import torch


def wrap_angle(theta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(theta), torch.cos(theta))


def apply_torsion_updates(
    coords: torch.Tensor,
    torsion_bonds: dict,
    torsion_masks: torch.Tensor,
    delta_angles: torch.Tensor,
) -> torch.Tensor:
    out = coords.clone()
    B, M, N = torsion_masks.shape
    j_idx = torsion_bonds["atom_j"]
    k_idx = torsion_bonds["atom_k"]

    for m in range(M):
        j = j_idx[:, m]
        k = k_idx[:, m]

        jpos = out[torch.arange(B, device=out.device), j]
        kpos = out[torch.arange(B, device=out.device), k]
        axis = kpos - jpos
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        a = delta_angles[:, m]
        ca = torch.cos(a).view(B, 1)
        sa = torch.sin(a).view(B, 1)

        mask_m = torsion_masks[:, m, :].unsqueeze(-1)
        rel = out - jpos.unsqueeze(1)

        term1 = rel * ca.unsqueeze(1)
        term2 = torch.cross(axis.unsqueeze(1).expand(-1, N, -1), rel, dim=-1) * sa.unsqueeze(1)
        proj = (rel * axis.unsqueeze(1)).sum(dim=-1, keepdim=True)
        term3 = axis.unsqueeze(1) * proj * (1.0 - ca).unsqueeze(1)
        rotated = term1 + term2 + term3

        updated = jpos.unsqueeze(1) + rotated
        out = torch.where(mask_m, updated, out)

    return out


def perturb_torsions(phi_0: torch.Tensor, sigma_tor: torch.Tensor, valid_mask: torch.Tensor):
    eps = torch.randn_like(phi_0)
    phi_t = wrap_angle(phi_0 + sigma_tor[:, None] * eps)
    delta = wrap_angle(phi_t - phi_0)
    target = -delta / sigma_tor[:, None].pow(2).clamp_min(1e-12)

    phi_t = torch.where(valid_mask, phi_t, phi_0)
    target = torch.where(valid_mask, target, torch.zeros_like(target))
    eps = torch.where(valid_mask, eps, torch.zeros_like(eps))
    return phi_t, eps, target


def torsion_score_loss(
    tor_score_pred: torch.Tensor,
    target_tor_score: torch.Tensor,
    sigma_tor: torch.Tensor,
    torsion_valid_mask: torch.Tensor,
) -> torch.Tensor:
    w = sigma_tor[:, None].pow(2)
    se = w * (tor_score_pred - target_tor_score).pow(2)
    se = se * torsion_valid_mask.float()
    denom = torsion_valid_mask.float().sum().clamp_min(1.0)
    return se.sum() / denom
