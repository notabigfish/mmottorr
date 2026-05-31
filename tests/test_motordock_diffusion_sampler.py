import torch

from motordock.diffusion.motordock_diffusion_sampler import sample_motordock_diffusion


class CountingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        B = batch["ligand_coords_t"].shape[0]
        M = batch.get("torsion_valid_mask", torch.zeros(B, 0, dtype=torch.bool)).shape[1]
        C = batch.get("pair_mask", torch.zeros(B, 0, dtype=torch.bool)).shape[1]
        device = batch["ligand_coords_t"].device
        dtype = batch["ligand_coords_t"].dtype
        return {
            "tr_score_pred": torch.zeros(B, 3, device=device, dtype=dtype),
            "rot_score_pred": torch.zeros(B, 3, device=device, dtype=dtype),
            "tor_score_pred": torch.zeros(B, M, device=device, dtype=dtype),
            "confidence_logit": torch.zeros(B, device=device, dtype=dtype),
            "contact_logit": torch.zeros(B, device=device, dtype=dtype),
            "pair_attention": torch.full((B, C), 1.0 / max(C, 1), device=device, dtype=dtype) if C > 0 else None,
            "pair_xi_pred": torch.zeros(B, C, 6, device=device, dtype=dtype) if C > 0 else None,
            "pair_T_corrected": torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4).repeat(B, C, 1, 1) if C > 0 else None,
            "selected_pair_xi": torch.zeros(B, 6, device=device, dtype=dtype),
        }


def _batch(B=2, A=7, P=12, C=3, M=2):
    return {
        "protein_ca": torch.randn(B, P, 3),
        "protein_feat": torch.randn(B, P, 16),
        "protein_mask": torch.ones(B, P, dtype=torch.bool),
        "pocket_mask": torch.ones(B, P, dtype=torch.long),
        "pocket_center": torch.randn(B, 3),
        "ligand_atom_feat": torch.randn(B, A, 10),
        "ligand_coords_start": torch.randn(B, A, 3),
        "ligand_mask": torch.ones(B, A, dtype=torch.bool),
        "pair_features": torch.randn(B, C, 8),
        "pair_mask": torch.ones(B, C, dtype=torch.bool),
        "pair_T_input": torch.eye(4).view(1, 1, 4, 4).repeat(B, C, 1, 1),
        "torsion_bond_atom_j": torch.zeros(B, M, dtype=torch.long),
        "torsion_bond_atom_k": torch.ones(B, M, dtype=torch.long),
        "torsion_valid_mask": torch.zeros(B, M, dtype=torch.bool),
    }


def test_motordock_sampler_iterative_call_count_and_shapes():
    model = CountingModel()
    b = _batch()
    out = sample_motordock_diffusion(model, b, num_samples=3, num_steps=5, ode=True)
    assert model.calls == 5
    assert out["ligand_coords_samples"].shape == (2, 3, 7, 3)
    assert out["candidate_score"].shape == (2, 3)
    assert out["top_ligand_coords"].shape == (2, 7, 3)
    assert out["top_sample_index"].shape == (2,)
    assert torch.isfinite(out["ligand_coords_samples"]).all()
