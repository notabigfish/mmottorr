import torch
from motordock.models.motordock_ablation_model import MotorDockAblationModel
from motordock.data.representation_pair_featurizer import representation_pair_feature_dim


def _batch(F):
    B,Np,Nl,C = 2,8,5,4
    return {
        "protein_feat": torch.randn(B,Np,24), "protein_ca": torch.randn(B,Np,3), "protein_mask": torch.ones(B,Np,dtype=torch.bool),
        "pocket_mask": torch.randint(0,2,(B,Np)), "domain_mask": torch.ones(B,Np,dtype=torch.long),
        "ligand_atom_feat": torch.randn(B,Nl,18), "ligand_coords_true": torch.randn(B,Nl,3), "ligand_coords_start": torch.randn(B,Nl,3), "ligand_mask": torch.ones(B,Nl,dtype=torch.bool),
        "pocket_center": torch.randn(B,3), "pair_features": torch.randn(B,C,F), "pair_mask": torch.ones(B,C,dtype=torch.bool),
        "pair_T_input": torch.eye(4).view(1,1,4,4).repeat(B,C,1,1), "pair_T_target_residual": torch.eye(4).view(1,1,4,4).repeat(B,C,1,1),
    }


def test_ablation_model_forward_all_representations():
    for r in ["se3_log","quaternion_translation","dual_quaternion","matrix","centroid_bias","random_motor","shuffled_pairs","no_pair_context","pga_feature"]:
        F = representation_pair_feature_dim(r)
        m = MotorDockAblationModel(24,18,F,representation=r,hidden_dim=32)
        o = m(_batch(F))
        assert "ligand_coords_pred" in o


def test_ablation_outputs_match_motordock_contract():
    r = "se3_log"; F = representation_pair_feature_dim(r)
    m = MotorDockAblationModel(24,18,F,representation=r,hidden_dim=32)
    o = m(_batch(F))
    assert o["xi_pred"].shape == (2,6)


def test_no_pair_context_forward_runs():
    r = "no_pair_context"; F = representation_pair_feature_dim(r)
    m = MotorDockAblationModel(24,18,F,representation=r,hidden_dim=32,disable_pair_context=True)
    o = m(_batch(F))
    assert o["confidence_logit"].shape[0] == 2


def test_outputs_are_finite():
    r = "dual_quaternion"; F = representation_pair_feature_dim(r)
    m = MotorDockAblationModel(24,18,F,representation=r,hidden_dim=32)
    o = m(_batch(F))
    assert torch.isfinite(o["ligand_coords_pred"]).all()
