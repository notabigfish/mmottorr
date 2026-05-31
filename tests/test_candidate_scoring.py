import torch

from motordock.scoring import score_candidates


def _make_inputs(B=3, P=10, A=6, C=4):
    batch = {
        "protein_ca": torch.randn(B, P, 3),
        "protein_mask": torch.ones(B, P, dtype=torch.bool),
        "pocket_mask": torch.ones(B, P, dtype=torch.long),
        "ligand_mask": torch.ones(B, A, dtype=torch.bool),
    }
    ligand_coords = torch.randn(B, A, 3)
    model_out = {
        "confidence_logit": torch.zeros(B),
        "contact_logit": torch.zeros(B),
        "pair_xi_pred": torch.zeros(B, C, 6),
        "pair_attention": torch.full((B, C), 1.0 / C),
    }
    return batch, model_out, ligand_coords


def test_candidate_scoring_shape_and_finite():
    batch, model_out, ligand_coords = _make_inputs()
    out = score_candidates(batch, model_out, ligand_coords)
    assert out["score"].shape == (3,)
    assert torch.isfinite(out["score"]).all()
    assert torch.isfinite(out["S_pose"]).all()
    assert torch.isfinite(out["S_contact"]).all()
    assert torch.isfinite(out["S_validity"]).all()


def test_candidate_scoring_confidence_increases_score():
    batch, model_out, ligand_coords = _make_inputs(B=2)
    model_out["confidence_logit"] = torch.tensor([0.0, 2.0])
    out = score_candidates(batch, model_out, ligand_coords)
    assert out["S_pose"][1] > out["S_pose"][0]
    assert out["score"][1] > out["score"][0]


def test_candidate_scoring_motor_penalty_and_clash_penalty():
    batch, model_out, ligand_coords = _make_inputs(B=2)
    # Larger motor norm for sample 1 should lower S_motor.
    model_out["pair_xi_pred"][1, :, :3] = 2.0
    out1 = score_candidates(batch, model_out, ligand_coords)
    assert out1["S_motor"][1] < out1["S_motor"][0]

    # Move ligand 0 close to protein to increase clashes and lower validity score.
    ligand_coords2 = ligand_coords.clone()
    ligand_coords2[0] = batch["protein_ca"][0, : ligand_coords2.shape[1]]
    out2 = score_candidates(batch, model_out, ligand_coords2)
    assert out2["E_clash"][0] >= out1["E_clash"][0]
    assert out2["S_validity"][0] <= out1["S_validity"][0]
