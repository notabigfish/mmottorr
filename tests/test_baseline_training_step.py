import torch
from motordock.models import BaselineDockingModel
from motordock.losses.pose_loss import rigid_docking_loss


def test_one_training_step_runs():
    B, Np, Nl = 2, 10, 6
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
        "T_target": torch.eye(4).unsqueeze(0).repeat(B, 1, 1),
    }
    m = BaselineDockingModel(24, 18)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    out = m(batch)
    ld = rigid_docking_loss(out, batch)
    loss = ld["total"]
    assert torch.isfinite(loss)
    opt.zero_grad()
    loss.backward()
    for p in m.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
    opt.step()
