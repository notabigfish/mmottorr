import torch
from motordock.models.motordock_se3_model import MotorDockSE3Model
from motordock.losses.motordock_loss import motordock_se3_loss


def test_one_motordock_training_step_runs():
    B, Np, Nl, C = 2, 10, 6, 4
    b = {
        "protein_feat": torch.randn(B, Np, 24), "protein_ca": torch.randn(B, Np, 3), "protein_mask": torch.ones(B, Np, dtype=torch.bool),
        "pocket_mask": torch.randint(0, 2, (B, Np)), "domain_mask": torch.ones(B, Np, dtype=torch.long),
        "ligand_atom_feat": torch.randn(B, Nl, 18), "ligand_coords_true": torch.randn(B, Nl, 3), "ligand_coords_start": torch.randn(B, Nl, 3), "ligand_mask": torch.ones(B, Nl, dtype=torch.bool),
        "pocket_center": torch.randn(B, 3), "T_target": torch.eye(4).view(1,4,4).repeat(B,1,1),
        "pair_features": torch.randn(B, C, 29), "pair_mask": torch.ones(B, C, dtype=torch.bool),
        "pair_T_input": torch.eye(4).view(1,1,4,4).repeat(B,C,1,1), "pair_T_target_residual": torch.eye(4).view(1,1,4,4).repeat(B,C,1,1),
        "pair_was_perturbed": torch.ones(B, C, dtype=torch.bool),
    }
    m = MotorDockSE3Model(24, 18, 29, hidden_dim=32)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    o = m(b)
    ld = motordock_se3_loss(o, b)
    assert torch.isfinite(ld["total"])
    opt.zero_grad(); ld["total"].backward(); opt.step()
