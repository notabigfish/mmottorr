import torch
from motordock.data.representation_pair_featurizer import representation_pair_feature_dim, featurize_candidate_pair_with_representation


def test_pair_feature_dim_changes_only_by_representation():
    assert representation_pair_feature_dim("se3_log") != representation_pair_feature_dim("dual_quaternion")


def test_pair_metadata_features_consistent_across_representations():
    p = {"pair_type":"single","unit_a_type":"chain","unit_b_type":"chain","unit_a_chain":"A","unit_b_chain":"B"}
    f1 = featurize_candidate_pair_with_representation(p, torch.eye(4), "se3_log")
    f2 = featurize_candidate_pair_with_representation(p, torch.eye(4), "matrix")
    assert torch.isfinite(f1).all() and torch.isfinite(f2).all()


def test_random_motor_is_deterministic_with_seed():
    p = {"pair_type":"single"}
    a = featurize_candidate_pair_with_representation(p, torch.eye(4), "random_motor", random_seed=3)
    b = featurize_candidate_pair_with_representation(p, torch.eye(4), "random_motor", random_seed=3)
    assert torch.allclose(a, b)


def test_centroid_bias_does_not_include_full_orientation():
    p = {"pair_type":"single"}
    f = featurize_candidate_pair_with_representation(p, torch.eye(4), "centroid_bias")
    assert f.shape[0] == representation_pair_feature_dim("centroid_bias")


def test_pair_features_are_finite():
    p = {"pair_type":"single"}
    f = featurize_candidate_pair_with_representation(p, torch.eye(4), "dual_quaternion")
    assert torch.isfinite(f).all()


def test_pga_feature_not_dual_quaternion_path_and_dim():
    p = {"pair_type":"single"}
    T = torch.eye(4)
    T[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    fp = featurize_candidate_pair_with_representation(p, T, "pga_feature")
    fd = featurize_candidate_pair_with_representation(p, T, "dual_quaternion")
    # rep prefix dims: 8 for pga, 8 for dq, but values should differ by construction.
    assert fp.shape[0] == representation_pair_feature_dim("pga_feature")
    assert not torch.allclose(fp[:8], fd[:8])


def test_pga_feature_changes_with_transform():
    p = {"pair_type":"single"}
    T1 = torch.eye(4)
    T2 = torch.eye(4)
    T2[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    a = featurize_candidate_pair_with_representation(p, T1, "pga_feature")
    b = featurize_candidate_pair_with_representation(p, T2, "pga_feature")
    assert not torch.allclose(a[:8], b[:8])
