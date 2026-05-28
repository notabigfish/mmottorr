import torch
from motordock.data.collate import baseline_collate_fn


def test_collate_pads_variable_lengths():
    b1 = {
        "protein_feat": torch.randn(3, 24), "protein_ca": torch.randn(3, 3), "protein_mask": torch.ones(3, dtype=torch.bool),
        "pocket_mask": torch.ones(3, dtype=torch.long), "domain_mask": torch.ones(3, dtype=torch.long),
        "ligand_atom_feat": torch.randn(4, 18), "ligand_coords_true": torch.randn(4, 3), "ligand_coords_start": torch.randn(4, 3), "ligand_mask": torch.ones(4, dtype=torch.bool),
        "pocket_center": torch.randn(3), "T_noise": torch.eye(4), "T_target": torch.eye(4), "affinity": 0.0,
        "pdb_id": "a", "motordock_type": "x", "ligand_file": "l", "protein_file": "p", "pocket_file": "k", "split": "train"
    }
    b2 = {
        **b1,
        "protein_feat": torch.randn(5, 24), "protein_ca": torch.randn(5, 3), "protein_mask": torch.ones(5, dtype=torch.bool),
        "pocket_mask": torch.ones(5, dtype=torch.long), "domain_mask": torch.ones(5, dtype=torch.long),
        "ligand_atom_feat": torch.randn(2, 18), "ligand_coords_true": torch.randn(2, 3), "ligand_coords_start": torch.randn(2, 3), "ligand_mask": torch.ones(2, dtype=torch.bool),
        "pdb_id": "b"
    }
    out = baseline_collate_fn([b1, b2])
    assert out["protein_feat"].shape[:2] == (2, 5)
    assert out["ligand_atom_feat"].shape[:2] == (2, 4)
