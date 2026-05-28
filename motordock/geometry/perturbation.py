from __future__ import annotations

import math
import torch

from .se3 import se3_exp_map


def random_rotation_vector(
    batch_shape=(),
    max_angle_degrees: float = 10.0,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Return random rotation vectors with norm <= max_angle_degrees."""
    max_angle = math.radians(max_angle_degrees)
    v = torch.randn(*batch_shape, 3, device=device, dtype=dtype)
    v = v / (torch.linalg.norm(v, dim=-1, keepdim=True) + 1e-8)
    mag = torch.rand(*batch_shape, 1, device=device, dtype=dtype) * max_angle
    return v * mag


def random_translation(
    batch_shape=(),
    max_translation: float = 2.0,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Return random translations with norm <= max_translation."""
    v = torch.randn(*batch_shape, 3, device=device, dtype=dtype)
    v = v / (torch.linalg.norm(v, dim=-1, keepdim=True) + 1e-8)
    mag = torch.rand(*batch_shape, 1, device=device, dtype=dtype) * float(max_translation)
    return v * mag


def perturb_transform(
    T: torch.Tensor,
    max_angle_degrees: float = 10.0,
    max_translation: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate xi perturbation and apply T_perturbed = exp(xi) @ T."""
    omega = random_rotation_vector(batch_shape=T.shape[:-2], max_angle_degrees=max_angle_degrees, device=T.device, dtype=T.dtype)
    trans = random_translation(batch_shape=T.shape[:-2], max_translation=max_translation, device=T.device, dtype=T.dtype)
    xi = torch.cat([omega, trans], dim=-1)
    T_delta = se3_exp_map(xi)
    return T_delta @ T, xi


def make_controlled_perturbations(
    T: torch.Tensor,
    angles_degrees: list[float] = [5.0, 10.0, 20.0],
    translations_angstrom: list[float] = [1.0, 2.0, 5.0],
) -> list[dict]:
    """Return deterministic perturbations for evaluation."""
    axis = torch.tensor([1.0, 0.0, 0.0], dtype=T.dtype, device=T.device)
    direction = torch.tensor([0.0, 1.0, 0.0], dtype=T.dtype, device=T.device)
    out = []
    for a in angles_degrees:
        for tr in translations_angstrom:
            omega = axis * math.radians(float(a))
            v = direction * float(tr)
            xi = torch.cat([omega, v], dim=-1)
            T_pert = se3_exp_map(xi) @ T
            out.append({
                "angle_degrees": float(a),
                "translation_angstrom": float(tr),
                "xi": xi,
                "T_perturbed": T_pert,
            })
    return out
