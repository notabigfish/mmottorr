from __future__ import annotations

import torch
import torch.nn.functional as F


def score_candidates(
    batch: dict,
    model_out: dict,
    ligand_coords: torch.Tensor,
    weights: dict | None = None,
    d_contact: float = 4.5,
    d_clash: float = 2.0,
    contact_tau: float = 0.5,
    clash_tau: float = 0.25,
    sigma_R: float = 0.2617993878,
    sigma_t: float = 2.0,
) -> dict:
    if weights is None:
        weights = {"pose": 1.0, "contact": 0.5, "motor": 0.25, "validity": 0.5}

    S_pose = model_out["confidence_logit"]

    protein_xyz = batch.get("protein_atom_coords", None)
    protein_mask = None
    if protein_xyz is None:
        protein_xyz = batch["protein_ca"]
        protein_mask = batch["protein_mask"]
    else:
        protein_mask = batch.get("protein_atom_mask")

    if "pocket_mask" in batch:
        protein_mask = protein_mask & (batch["pocket_mask"] > 0)

    d = torch.cdist(protein_xyz, ligand_coords)
    contact = torch.sigmoid((d_contact - d) / contact_tau)
    mask = protein_mask[:, :, None] & batch["ligand_mask"][:, None, :]
    mask_f = mask.float()
    S_contact_geom = (contact * mask_f).sum(dim=(1, 2)) / mask_f.sum(dim=(1, 2)).clamp_min(1.0)

    if "contact_logit" in model_out and model_out["contact_logit"] is not None:
        S_contact = model_out["contact_logit"] + torch.logit(S_contact_geom.clamp(1e-4, 1 - 1e-4))
    else:
        S_contact = torch.logit(S_contact_geom.clamp(1e-4, 1 - 1e-4))

    xi = model_out.get("selected_pair_xi", None)
    if xi is None:
        pair_xi = model_out.get("pair_xi_pred", None)
        pair_attn = model_out.get("pair_attention", None)
        if pair_xi is not None and pair_attn is not None:
            xi = (pair_xi * pair_attn.unsqueeze(-1)).sum(dim=1)

    if xi is None:
        E_motor = torch.zeros_like(S_pose)
        S_motor = torch.zeros_like(S_pose)
    else:
        omega = xi[:, :3]
        v = xi[:, 3:]
        E_motor = (omega.square().sum(dim=-1) / (sigma_R**2)) + (v.square().sum(dim=-1) / (sigma_t**2))
        S_motor = -E_motor

    clash = F.softplus((d_clash - d) / clash_tau)
    E_clash = (clash * mask_f).sum(dim=(1, 2)) / mask_f.sum(dim=(1, 2)).clamp_min(1.0)

    E_bond = torch.zeros_like(E_clash)
    if all(k in batch for k in ["ligand_bond_index", "ligand_bond_length_ref", "ligand_bond_mask"]):
        idx = batch["ligand_bond_index"]  # [B,E,2]
        i = idx[:, :, 0]
        j = idx[:, :, 1]
        li = torch.gather(ligand_coords, 1, i.unsqueeze(-1).expand(-1, -1, 3))
        lj = torch.gather(ligand_coords, 1, j.unsqueeze(-1).expand(-1, -1, 3))
        le = torch.linalg.norm(li - lj, dim=-1)
        ref = batch["ligand_bond_length_ref"]
        bmask = batch["ligand_bond_mask"].float()
        E_bond = ((le - ref).square() * bmask).sum(dim=1) / bmask.sum(dim=1).clamp_min(1.0)

    S_validity = -E_clash - E_bond

    S = (
        weights["pose"] * S_pose
        + weights["contact"] * S_contact
        + weights["motor"] * S_motor
        + weights["validity"] * S_validity
    )

    return {
        "score": S,
        "S_pose": S_pose,
        "S_contact": S_contact,
        "S_motor": S_motor,
        "S_validity": S_validity,
        "contact_geom": S_contact_geom,
        "E_motor": E_motor,
        "E_clash": E_clash,
        "E_bond": E_bond,
    }
