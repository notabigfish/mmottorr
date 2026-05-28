import torch
from motordock.models.motordock_ablation_model import MotorDockAblationModel
from motordock.losses.ablation_loss import motordock_ablation_loss
from motordock.data.representation_pair_featurizer import representation_pair_feature_dim


def _batch(F):
    B,Np,Nl,C = 2,8,5,4
    return {
        "protein_feat": torch.randn(B,Np,24), "protein_ca": torch.randn(B,Np,3), "protein_mask": torch.ones(B,Np,dtype=torch.bool),
        "pocket_mask": torch.randint(0,2,(B,Np)), "domain_mask": torch.ones(B,Np,dtype=torch.long),
        "ligand_atom_feat": torch.randn(B,Nl,18), "ligand_coords_true": torch.randn(B,Nl,3), "ligand_coords_start": torch.randn(B,Nl,3), "ligand_mask": torch.ones(B,Nl,dtype=torch.bool),
        "pocket_center": torch.randn(B,3), "T_target": torch.eye(4).view(1,4,4).repeat(B,1,1),
        "pair_features": torch.randn(B,C,F), "pair_mask": torch.ones(B,C,dtype=torch.bool),
        "pair_T_input": torch.eye(4).view(1,1,4,4).repeat(B,C,1,1), "pair_T_target_residual": torch.eye(4).view(1,1,4,4).repeat(B,C,1,1), "pair_was_perturbed": torch.ones(B,C,dtype=torch.bool),
    }


def test_one_training_step_each_representation():
    reps = ["se3_log","quaternion_translation","dual_quaternion","matrix","centroid_bias","random_motor","shuffled_pairs","no_pair_context","pga_feature"]
    for r in reps:
        F = representation_pair_feature_dim(r)
        m = MotorDockAblationModel(24,18,F,representation=r,hidden_dim=32,disable_pair_context=(r=="no_pair_context"))
        b = _batch(F)
        o = m(b)
        l = motordock_ablation_loss(o,b,r)
        assert torch.isfinite(l["total"])
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        opt.zero_grad(); l["total"].backward(); opt.step()
