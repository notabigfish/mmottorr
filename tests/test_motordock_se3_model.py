import torch
from motordock.models.motordock_se3_model import MotorDockSE3Model


def _batch(B=2, Np=8, Nl=5, C=4):
    return {
        "protein_feat": torch.randn(B, Np, 24), "protein_ca": torch.randn(B, Np, 3), "protein_mask": torch.ones(B, Np, dtype=torch.bool),
        "pocket_mask": torch.randint(0, 2, (B, Np)), "domain_mask": torch.ones(B, Np, dtype=torch.long),
        "ligand_atom_feat": torch.randn(B, Nl, 18), "ligand_coords_true": torch.randn(B, Nl, 3), "ligand_coords_start": torch.randn(B, Nl, 3), "ligand_mask": torch.ones(B, Nl, dtype=torch.bool),
        "pocket_center": torch.randn(B, 3), "pair_features": torch.randn(B, C, 29), "pair_mask": torch.ones(B, C, dtype=torch.bool),
        "pair_T_input": torch.eye(4).view(1,1,4,4).repeat(B,C,1,1), "pair_T_target_residual": torch.eye(4).view(1,1,4,4).repeat(B,C,1,1),
    }


def test_motordock_forward_shapes():
    m = MotorDockSE3Model(24, 18, 29, hidden_dim=32)
    o = m(_batch())
    assert o["xi_pred"].shape == (2,6)
    assert o["pair_xi_pred"].shape[2] == 6


def test_motordock_forward_outputs_finite():
    m = MotorDockSE3Model(24, 18, 29, hidden_dim=32)
    o = m(_batch())
    assert torch.isfinite(o["ligand_coords_pred"]).all()


def test_pair_attention_sums_to_one_on_valid_pairs():
    m = MotorDockSE3Model(24, 18, 29, hidden_dim=32)
    o = m(_batch())
    s = o["pair_attention"].sum(dim=1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)


def test_disable_pair_context_forward_still_runs():
    m = MotorDockSE3Model(24, 18, 29, hidden_dim=32, disable_pair_context=True)
    o = m(_batch())
    assert o["confidence_logit"].shape[0] == 2
