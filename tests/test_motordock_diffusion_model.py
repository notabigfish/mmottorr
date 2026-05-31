import torch

from motordock.models import MotorDockDiffusionModel


def _batch(B=2, P=12, A=7, C=3, Fp=16, Fl=10, Fpair=8, M=4):
    eye = torch.eye(4).view(1, 1, 4, 4).repeat(B, C, 1, 1)
    return {
        "protein_feat": torch.randn(B, P, Fp),
        "protein_ca": torch.randn(B, P, 3),
        "protein_mask": torch.ones(B, P, dtype=torch.bool),
        "pocket_mask": torch.ones(B, P, dtype=torch.long),
        "pocket_center": torch.randn(B, 3),
        "ligand_atom_feat": torch.randn(B, A, Fl),
        "ligand_mask": torch.ones(B, A, dtype=torch.bool),
        "ligand_coords_t": torch.randn(B, A, 3),
        "ligand_coords_start": torch.randn(B, A, 3),
        "sigma_tr": torch.full((B,), 1.0),
        "sigma_rot": torch.full((B,), 0.5),
        "pair_features": torch.randn(B, C, Fpair),
        "pair_mask": torch.ones(B, C, dtype=torch.bool),
        "pair_T_input": eye.clone(),
        "torsion_bond_atom_j": torch.zeros(B, M, dtype=torch.long),
        "torsion_bond_atom_k": torch.ones(B, M, dtype=torch.long),
        "torsion_valid_mask": torch.ones(B, M, dtype=torch.bool),
    }


def test_motordock_diffusion_forward_shapes():
    b = _batch()
    m = MotorDockDiffusionModel(16, 10, 8, hidden_dim=32, sigma_emb_dim=16, adapter_type="se3")
    o = m(b)
    assert o["tr_score_pred"].shape == (2, 3)
    assert o["rot_score_pred"].shape == (2, 3)
    assert o["tor_score_pred"].shape == (2, 4)
    assert o["confidence_logit"].shape == (2,)
    assert o["contact_logit"].shape == (2,)
    assert o["pair_attention"].shape == (2, 3)
    assert o["pair_xi_pred"].shape == (2, 3, 6)
    assert o["pair_T_corrected"].shape == (2, 3, 4, 4)
    assert torch.isfinite(o["tr_score_pred"]).all()
    assert torch.isfinite(o["rot_score_pred"]).all()


def test_motordock_diffusion_no_candidate_pairs():
    b = _batch()
    b["pair_mask"] = torch.zeros_like(b["pair_mask"])
    m = MotorDockDiffusionModel(16, 10, 8, hidden_dim=32, sigma_emb_dim=16, adapter_type="se3")
    o = m(b)
    assert o["tr_score_pred"].shape == (2, 3)
    assert torch.isfinite(o["tr_score_pred"]).all()
    assert torch.isfinite(o["rot_score_pred"]).all()
    assert torch.isfinite(o["confidence_logit"]).all()


def test_motordock_diffusion_pga_sandwich_forward_runs():
    b = _batch()
    m = MotorDockDiffusionModel(16, 10, 8, hidden_dim=32, sigma_emb_dim=16, adapter_type="pga_sandwich")
    o = m(b)
    assert o["pair_T_corrected"].shape == (2, 3, 4, 4)
    assert torch.isfinite(o["pair_T_corrected"]).all()
