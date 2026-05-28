import torch
from motordock.models import BaselineDockingModel


def test_model_forward_shapes():
    B, Np, Nl = 2, 8, 5
    batch = {
        "protein_feat": torch.randn(B, Np, 24),
        "protein_ca": torch.randn(B, Np, 3),
        "protein_mask": torch.ones(B, Np, dtype=torch.bool),
        "pocket_mask": torch.randint(0, 2, (B, Np)),
        "domain_mask": torch.ones(B, Np, dtype=torch.long),
        "ligand_atom_feat": torch.randn(B, Nl, 18),
        "ligand_coords_true": torch.randn(B, Nl, 3),
        "ligand_coords_start": torch.randn(B, Nl, 3),
        "ligand_mask": torch.ones(B, Nl, dtype=torch.bool),
        "pocket_center": torch.randn(B, 3),
    }
    m = BaselineDockingModel(24, 18)
    o = m(batch)
    assert o["xi_pred"].shape == (B, 6)
    assert o["delta_T_pred"].shape == (B, 4, 4)
    assert o["ligand_coords_pred"].shape == (B, Nl, 3)
    assert o["confidence_logit"].shape == (B,)
    assert torch.isfinite(o["ligand_coords_pred"]).all()
