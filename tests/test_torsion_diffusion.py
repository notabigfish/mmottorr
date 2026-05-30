import torch

from motordock.diffusion.noise_schedule import DiffusionSchedule
from motordock.diffusion.torsion import torsion_score_loss
from motordock.models.pose_sampler import DiffusionPoseSampler


class DummyModelWithTorsion(torch.nn.Module):
    def __init__(self, tor_value: float):
        super().__init__()
        self.tor_value = tor_value

    def forward(self, batch):
        B = batch["ligand_coords_t"].shape[0]
        M = batch["torsion_valid_mask"].shape[1]
        device = batch["ligand_coords_t"].device
        dtype = batch["ligand_coords_t"].dtype
        return {
            "tr_score_pred": torch.zeros(B, 3, device=device, dtype=dtype),
            "rot_score_pred": torch.zeros(B, 3, device=device, dtype=dtype),
            "tor_score_pred": torch.full((B, M), self.tor_value, device=device, dtype=dtype),
            "confidence_logit": torch.zeros(B, device=device, dtype=dtype),
        }


def test_torsion_loss_ignores_padding():
    B, M = 2, 4
    pred = torch.tensor([[1.0, 2.0, 99.0, 99.0], [1.0, 3.0, 99.0, 99.0]])
    tgt = torch.tensor([[0.0, 2.0, -50.0, -50.0], [0.0, 1.0, -50.0, -50.0]])
    sigma = torch.ones(B)
    valid = torch.tensor([[True, True, False, False], [True, True, False, False]])

    l_full = torsion_score_loss(pred, tgt, sigma, valid)
    l_expected = (((pred[:, :2] - tgt[:, :2]) ** 2).sum()) / 4.0
    assert torch.allclose(l_full, l_expected)


def _batch_with_one_torsion(B=1, N=6, Np=8):
    coords = torch.randn(B, N, 3)
    mask = torch.ones(B, N, dtype=torch.bool)
    torsion_atom_mask = torch.zeros(B, 1, N, dtype=torch.bool)
    torsion_atom_mask[:, 0, N // 2 :] = True
    return {
        "protein_ca": torch.randn(B, Np, 3),
        "protein_feat": torch.randn(B, Np, 24),
        "protein_mask": torch.ones(B, Np, dtype=torch.bool),
        "pocket_mask": torch.ones(B, Np, dtype=torch.long),
        "ligand_atom_feat": torch.randn(B, N, 18),
        "ligand_coords_start": coords,
        "ligand_mask": mask,
        "pocket_center": torch.randn(B, 3),
        "torsion_bond_atom_j": torch.tensor([[1]], dtype=torch.long),
        "torsion_bond_atom_k": torch.tensor([[2]], dtype=torch.long),
        "torsion_atom_mask": torsion_atom_mask,
        "torsion_valid_mask": torch.tensor([[True]], dtype=torch.bool),
        "torsion_angles_0": torch.zeros(B, 1),
    }


def test_sampler_with_torsions_changes_coords_vs_rigid_only():
    batch = _batch_with_one_torsion()
    sched = DiffusionSchedule(4, 0.1, 0.5, 0.05, 0.2)

    rigid_model = DummyModelWithTorsion(tor_value=0.0)
    tor_model = DummyModelWithTorsion(tor_value=0.8)

    s_rigid = DiffusionPoseSampler(rigid_model, sched, num_samples=1, deterministic=True)
    s_tor = DiffusionPoseSampler(tor_model, sched, num_samples=1, deterministic=True)

    out_rigid = s_rigid.sample(batch)["coords"][:, 0]
    out_tor = s_tor.sample(batch)["coords"][:, 0]

    assert not torch.allclose(out_rigid, out_tor)
