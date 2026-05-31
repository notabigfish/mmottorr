from __future__ import annotations

import torch

from motordock.geometry.se3 import so3_exp_map
from motordock.diffusion.rigid_pose import center_of_mass
from motordock.diffusion.torsion import apply_torsion_updates
from motordock.scoring import score_candidates


def geometric_sigma_schedule(sigma_max, sigma_min, num_steps, device, dtype):
    ratio = sigma_min / sigma_max
    t = torch.linspace(0, 1, num_steps, device=device, dtype=dtype)
    return sigma_max * (ratio ** t)


def _repeat_batch(batch: dict, num_samples: int) -> tuple[dict, int]:
    B = batch["ligand_coords_start"].shape[0]
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.shape[0] == B:
            out[k] = v.repeat_interleave(num_samples, dim=0)
        elif isinstance(v, list) and len(v) == B:
            out[k] = [x for x in v for _ in range(num_samples)]
        else:
            out[k] = v
    return out, B


def _random_rotation_matrix(B, device, dtype):
    omega = torch.randn(B, 3, device=device, dtype=dtype)
    return so3_exp_map(omega)


@torch.no_grad()
def sample_motordock_diffusion(
    model: torch.nn.Module,
    batch: dict,
    num_samples: int = 20,
    num_steps: int = 20,
    sigma_tr_max: float = 10.0,
    sigma_tr_min: float = 0.1,
    sigma_rot_max: float = 1.5,
    sigma_rot_min: float = 0.05,
    sigma_tor_max: float = 3.14,
    sigma_tor_min: float = 0.05,
    ode: bool = False,
    temperature: float = 1.0,
    candidate_score_weights: dict | None = None,
    return_trajectory: bool = False,
) -> dict:
    model.eval()

    work_batch, B = _repeat_batch(batch, num_samples)
    B2 = B * num_samples
    device = work_batch["ligand_coords_start"].device
    dtype = work_batch["ligand_coords_start"].dtype

    coords0 = work_batch["ligand_coords_start"]
    lig_mask = work_batch["ligand_mask"]
    lig_centroid = center_of_mass(coords0, lig_mask)
    coords_centered = coords0 - lig_centroid[:, None, :]

    R0 = _random_rotation_matrix(B2, device=device, dtype=dtype)
    coords_rot = torch.einsum("bij,baj->bai", R0, coords_centered)

    eps = torch.randn(B2, 3, device=device, dtype=dtype) * sigma_tr_max
    coords_t = coords_rot + work_batch["pocket_center"][:, None, :] + eps[:, None, :]

    sched_tr = geometric_sigma_schedule(sigma_tr_max, sigma_tr_min, num_steps, device, dtype)
    sched_rot = geometric_sigma_schedule(sigma_rot_max, sigma_rot_min, num_steps, device, dtype)
    sched_tor = geometric_sigma_schedule(sigma_tor_max, sigma_tor_min, num_steps, device, dtype)

    traj = []
    if return_trajectory:
        traj.append(coords_t.clone())

    last_out = None
    for s in range(num_steps):
        sigma_tr = sched_tr[s]
        sigma_rot = sched_rot[s]
        sigma_tor = sched_tor[s]
        sigma_tr_next = sched_tr[s + 1] if s + 1 < num_steps else torch.tensor(sigma_tr_min, device=device, dtype=dtype)
        sigma_rot_next = sched_rot[s + 1] if s + 1 < num_steps else torch.tensor(sigma_rot_min, device=device, dtype=dtype)
        sigma_tor_next = sched_tor[s + 1] if s + 1 < num_steps else torch.tensor(sigma_tor_min, device=device, dtype=dtype)

        work_batch["ligand_coords_t"] = coords_t
        work_batch["sigma_tr"] = torch.full((B2,), float(sigma_tr.item()), device=device, dtype=dtype)
        work_batch["sigma_rot"] = torch.full((B2,), float(sigma_rot.item()), device=device, dtype=dtype)
        work_batch["sigma_tor"] = torch.full((B2,), float(sigma_tor.item()), device=device, dtype=dtype)

        out = model(work_batch)
        last_out = out

        # translation
        dt_tr = sigma_tr**2 - sigma_tr_next**2
        delta_tr = dt_tr * out["tr_score_pred"]
        if not ode and s + 1 < num_steps:
            noise = torch.randn_like(delta_tr) * temperature
            noise_scale = torch.sqrt(torch.clamp(sigma_tr_next**2 - sigma_tr_min**2, min=0.0))
            delta_tr = delta_tr + noise_scale * noise
        coords_t = coords_t + delta_tr[:, None, :]

        # rotation
        dt_rot = sigma_rot**2 - sigma_rot_next**2
        delta_omega = dt_rot * out["rot_score_pred"]
        if not ode and s + 1 < num_steps:
            delta_omega = delta_omega + torch.randn_like(delta_omega) * sigma_rot_next * temperature
        R_delta = so3_exp_map(delta_omega)
        center = center_of_mass(coords_t, work_batch["ligand_mask"])
        coords_t = torch.einsum("bij,baj->bai", R_delta, coords_t - center[:, None, :]) + center[:, None, :]

        # torsion
        if "tor_score_pred" in out and "torsion_valid_mask" in work_batch and work_batch["torsion_valid_mask"].shape[1] > 0:
            dt_tor = sigma_tor**2 - sigma_tor_next**2
            delta_tor = dt_tor * out["tor_score_pred"]
            if not ode and s + 1 < num_steps:
                delta_tor = delta_tor + torch.randn_like(delta_tor) * sigma_tor_next * temperature
            if "torsion_atom_mask" in work_batch:
                coords_t = apply_torsion_updates(
                    coords_t,
                    torsion_bonds={
                        "atom_j": work_batch["torsion_bond_atom_j"],
                        "atom_k": work_batch["torsion_bond_atom_k"],
                    },
                    torsion_masks=work_batch["torsion_atom_mask"],
                    delta_angles=delta_tor,
                )

        if return_trajectory:
            traj.append(coords_t.clone())

    score_out = score_candidates(work_batch, last_out, coords_t, weights=candidate_score_weights)
    score = score_out["score"]

    coords_samples = coords_t.view(B, num_samples, *coords_t.shape[1:])
    score_samples = score.view(B, num_samples)
    top_idx = torch.argmax(score_samples, dim=1)
    top_coords = coords_samples[torch.arange(B, device=device), top_idx]

    out = {
        "ligand_coords_samples": coords_samples,
        "candidate_score": score_samples,
        "score_components": {k: v.view(B, num_samples) if torch.is_tensor(v) and v.shape[0] == B2 else v for k, v in score_out.items() if k != "score"},
        "confidence_logit": last_out["confidence_logit"].view(B, num_samples),
        "contact_logit": last_out.get("contact_logit", torch.zeros(B2, device=device, dtype=dtype)).view(B, num_samples),
        "pair_attention": None if last_out.get("pair_attention", None) is None else last_out["pair_attention"].view(B, num_samples, -1),
        "pair_xi_pred": None if last_out.get("pair_xi_pred", None) is None else last_out["pair_xi_pred"].view(B, num_samples, last_out["pair_xi_pred"].shape[1], 6),
        "pair_T_corrected": None if last_out.get("pair_T_corrected", None) is None else last_out["pair_T_corrected"].view(B, num_samples, last_out["pair_T_corrected"].shape[1], 4, 4),
        "top_ligand_coords": top_coords,
        "top_sample_index": top_idx,
    }
    if return_trajectory:
        out["trajectory"] = [t.view(B, num_samples, *t.shape[1:]) for t in traj]
    return out
