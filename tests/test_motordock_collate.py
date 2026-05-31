import torch
from motordock.data.motordock_collate import motordock_se3_collate_fn


def _sample(c):
    return {
        "protein_feat": torch.randn(4, 24), "protein_ca": torch.randn(4, 3), "protein_mask": torch.ones(4, dtype=torch.bool),
        "pocket_mask": torch.ones(4, dtype=torch.long), "domain_mask": torch.ones(4, dtype=torch.long),
        "ligand_atom_feat": torch.randn(3, 18), "ligand_coords_true": torch.randn(3, 3), "ligand_coords_start": torch.randn(3, 3), "ligand_mask": torch.ones(3, dtype=torch.bool),
        "pocket_center": torch.randn(3), "T_noise": torch.eye(4), "T_target": torch.eye(4), "affinity": 0.0,
        "pdb_id": "x", "motordock_type": "Single-Domain", "ligand_file": "l", "protein_file": "p", "pocket_file": "k", "split": "train",
        "pair_features": torch.randn(c, 29), "pair_mask": torch.ones(c, dtype=torch.bool),
        "pair_T_input": torch.eye(4).unsqueeze(0).repeat(c, 1, 1), "pair_T_native": torch.eye(4).unsqueeze(0).repeat(c, 1, 1),
        "pair_T_target_residual": torch.eye(4).unsqueeze(0).repeat(c, 1, 1), "pair_valid": torch.ones(c, dtype=torch.bool),
        "pair_was_perturbed": torch.zeros(c, dtype=torch.bool), "pair_ids": [f"id{i}" for i in range(c)], "pair_types": ["single"] * c,
        "torsion_bond_atom_j": torch.zeros(0, dtype=torch.long),
        "torsion_bond_atom_k": torch.zeros(0, dtype=torch.long),
        "torsion_atom_mask": torch.zeros(0, 4, dtype=torch.bool), 
        "torsion_valid_mask": torch.zeros(0, dtype=torch.bool),
        "torsion_angles_0": torch.zeros(0, dtype=torch.float32)
    }


def test_motordock_collate_pads_candidate_pairs():
    b = motordock_se3_collate_fn([_sample(2), _sample(5)])
    assert b["pair_features"].shape[:2] == (2, 5)


def test_motordock_collate_pads_transforms_with_identity():
    b = motordock_se3_collate_fn([_sample(1), _sample(3)])
    assert torch.allclose(b["pair_T_input"][0, 2], torch.eye(4))


def test_motordock_collate_preserves_metadata_lists():
    b = motordock_se3_collate_fn([_sample(2), _sample(3)])
    assert isinstance(b["pair_ids"], list)
