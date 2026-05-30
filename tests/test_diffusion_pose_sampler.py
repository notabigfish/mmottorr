import torch

from motordock.diffusion.noise_schedule import DiffusionSchedule
from motordock.models.pose_sampler import DiffusionPoseSampler
from motordock.diffusion.rigid_pose import center_of_mass


class ZeroScoreModel(torch.nn.Module):
    def forward(self, batch):
        B = batch["ligand_coords_t"].shape[0]
        device = batch["ligand_coords_t"].device
        dtype = batch["ligand_coords_t"].dtype
        return {
            "tr_score_pred": torch.zeros(B, 3, device=device, dtype=dtype),
            "rot_score_pred": torch.zeros(B, 3, device=device, dtype=dtype),
            "confidence_logit": torch.zeros(B, device=device, dtype=dtype),
        }


class CentroidRestoreModel(torch.nn.Module):
    def forward(self, batch):
        x = batch["ligand_coords_t"]
        m = batch["ligand_mask"]
        c = center_of_mass(x, m)
        p = batch["pocket_center"]
        sigma_tr = batch["sigma_tr"].view(-1, 1)
        sigma_rot = batch["sigma_rot"].view(-1, 1)

        # choose score so deterministic update moves centroid toward pocket center
        tr_score = (p - c) / (sigma_tr.pow(2).clamp_min(1e-8))
        rot_score = torch.zeros_like(tr_score) / sigma_rot.clamp_min(1e-8)
        conf = -torch.linalg.norm(c - p, dim=-1)
        return {
            "tr_score_pred": tr_score,
            "rot_score_pred": rot_score,
            "confidence_logit": conf,
        }


def _fake_batch(B=2, Nl=5, Np=8):
    return {
        "protein_ca": torch.randn(B, Np, 3),
        "protein_feat": torch.randn(B, Np, 24),
        "protein_mask": torch.ones(B, Np, dtype=torch.bool),
        "pocket_mask": torch.ones(B, Np, dtype=torch.long),
        "ligand_atom_feat": torch.randn(B, Nl, 18),
        "ligand_coords_start": torch.randn(B, Nl, 3),
        "ligand_mask": torch.ones(B, Nl, dtype=torch.bool),
        "pocket_center": torch.randn(B, 3),
    }


def test_sampler_shape_cpu():
    batch = _fake_batch(B=2, Nl=5, Np=8)
    model = ZeroScoreModel()
    sched = DiffusionSchedule(3, 0.1, 2.0, 0.05, 1.0)
    sampler = DiffusionPoseSampler(model, sched, num_samples=4, deterministic=False)
    out = sampler.sample(batch)

    assert out["coords"].shape == (2, 4, 5, 3)
    assert out["ranked_indices"].shape == (2, 4)
    assert torch.is_tensor(out["coords"])
    assert not isinstance(out["coords"], range)


def test_exact_score_denoising_reduces_centroid_distance():
    B, N = 2, 5
    batch = _fake_batch(B=B, Nl=N, Np=8)
    batch["ligand_coords_start"] = batch["ligand_coords_start"] + 5.0
    model = CentroidRestoreModel()
    sched = DiffusionSchedule(8, 0.1, 1.0, 0.05, 0.5)
    sampler = DiffusionPoseSampler(model, sched, num_samples=1, deterministic=True)

    init_center = center_of_mass(batch["ligand_coords_start"], batch["ligand_mask"])
    init_dist = torch.linalg.norm(init_center - batch["pocket_center"], dim=-1).mean()

    out = sampler.sample(batch)
    fin = out["coords"][:, 0]
    fin_center = center_of_mass(fin, batch["ligand_mask"])
    fin_dist = torch.linalg.norm(fin_center - batch["pocket_center"], dim=-1).mean()

    assert fin_dist < init_dist
