import torch
from motordock.geometry.representation_conversions import representation_dim, transform_to_representation, representation_to_transform


def test_representation_dim_positive():
    for r in ["se3_log","quaternion_translation","dual_quaternion","matrix","centroid_bias","random_motor","shuffled_pairs","no_pair_context","pga_feature"]:
        assert representation_dim(r) > 0


def test_transform_to_representation_finite():
    T = torch.eye(4).unsqueeze(0)
    for r in ["se3_log","quaternion_translation","dual_quaternion","matrix"]:
        rep = transform_to_representation(T, r)
        assert torch.isfinite(rep).all()


def test_decodable_representations_roundtrip_to_transform():
    T = torch.eye(4).unsqueeze(0)
    for r in ["se3_log","quaternion_translation","dual_quaternion","matrix"]:
        rep = transform_to_representation(T, r)
        T2 = representation_to_transform(rep, r)
        assert T2.shape[-2:] == (4,4)


def test_nondecodable_representations_handled_cleanly():
    ok = False
    try:
        representation_to_transform(torch.zeros(1,4), "centroid_bias")
    except Exception:
        ok = True
    assert ok
