import torch

from motordock.diffusion.rigid_pose import perturb_rigid_pose, prepare_diffusion_batch_targets
from motordock.models.diffusion_docking_model import DiffusionDockingModel
from motordock.losses.pose_loss import diffusion_rigid_loss


def _batch(B=3, Nl=6, Np=8):
    return {
        "protein_ca": torch.randn(B, Np, 3),
        "protein_feat": torch.randn(B, Np, 24),
        "protein_mask": torch.ones(B, Np, dtype=torch.bool),
        "pocket_mask": torch.ones(B, Np, dtype=torch.long),
        "ligand_atom_feat": torch.randn(B, Nl, 18),
        "ligand_coords_true": torch.randn(B, Nl, 3),
        "ligand_mask": torch.ones(B, Nl, dtype=torch.bool),
        "pocket_center": torch.randn(B, 3),
    }


def test_training_targets_finite():
    b = _batch(B=4)
    for s in [1e-3, 1e-2, 1e-1, 1.0]:
        sigma_tr = torch.full((4,), s)
        sigma_rot = torch.full((4,), s)
        _, _, _, tr_tgt, rot_tgt = perturb_rigid_pose(
            b["ligand_coords_true"],
            b["ligand_mask"],
            b["pocket_center"],
            sigma_tr,
            sigma_rot,
        )
        assert torch.isfinite(tr_tgt).all()
        assert torch.isfinite(rot_tgt).all()


def test_gradient_flow_diffusion_model():
    b = _batch(B=2)
    sigma_tr = torch.full((2,), 0.5)
    sigma_rot = torch.full((2,), 0.3)
    bt = prepare_diffusion_batch_targets(b, sigma_tr, sigma_rot)

    model = DiffusionDockingModel(24, 18, hidden_dim=64, num_layers=2, dropout=0.0)
    out = model(bt)
    loss_dict = diffusion_rigid_loss(out, bt, bt["sigma_tr"], bt["sigma_rot"])
    loss = loss_dict["total"]
    assert torch.isfinite(loss)

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(torch.any(g.abs() > 0) for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
