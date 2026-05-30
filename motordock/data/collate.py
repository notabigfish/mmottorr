from __future__ import annotations

import torch


def _pad_2d(tensors, value=0.0):
    b = len(tensors)
    nmax = max(t.shape[0] for t in tensors)
    f = tensors[0].shape[1]
    out = torch.full((b, nmax, f), value, dtype=tensors[0].dtype)
    mask = torch.zeros((b, nmax), dtype=torch.bool)
    for i, t in enumerate(tensors):
        n = t.shape[0]
        out[i, :n] = t
        mask[i, :n] = True
    return out, mask


def _pad_1d(tensors, dtype=torch.float32):
    b = len(tensors)
    nmax = max(t.shape[0] for t in tensors)
    out = torch.zeros((b, nmax), dtype=dtype)
    for i, t in enumerate(tensors):
        out[i, : t.shape[0]] = t.to(dtype)
    return out


def baseline_collate_fn(batch: list[dict]) -> dict:
    protein_feat, protein_mask = _pad_2d([x["protein_feat"] for x in batch])
    protein_ca, _ = _pad_2d([x["protein_ca"] for x in batch])
    ligand_feat, ligand_mask = _pad_2d([x["ligand_atom_feat"] for x in batch])
    ligand_true, _ = _pad_2d([x["ligand_coords_true"] for x in batch])
    ligand_start, _ = _pad_2d([x["ligand_coords_start"] for x in batch])

    return {
        "protein_feat": protein_feat,
        "protein_ca": protein_ca,
        "protein_mask": protein_mask,
        "pocket_mask": _pad_1d([x["pocket_mask"] for x in batch], dtype=torch.long),
        "domain_mask": _pad_1d([x["domain_mask"] for x in batch], dtype=torch.long),
        "ligand_atom_feat": ligand_feat,
        "ligand_coords_true": ligand_true,
        "ligand_coords_start": ligand_start,
        "ligand_mask": ligand_mask,
        "pocket_center": torch.stack([x["pocket_center"] for x in batch], dim=0),
        "T_noise": torch.stack([x["T_noise"] for x in batch], dim=0),
        "T_target": torch.stack([x["T_target"] for x in batch], dim=0),
        "affinity": torch.tensor([x["affinity"] for x in batch], dtype=torch.float32),
        "pdb_id": [x["pdb_id"] for x in batch],
        "motordock_type": [x["motordock_type"] for x in batch],
        "ligand_file": [x["ligand_file"] for x in batch],
        "protein_file": [x["protein_file"] for x in batch],
        "pocket_file": [x["pocket_file"] for x in batch],
        "split": [x["split"] for x in batch],
    }
