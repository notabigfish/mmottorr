import torch
from motordock.geometry.matrix_representation import transform_to_matrix_features, matrix_features_to_transform
from motordock.geometry.se3 import is_valid_rotation


def test_matrix_3x4_roundtrip():
    T = torch.eye(4)
    f = transform_to_matrix_features(T, "3x4")
    T2 = matrix_features_to_transform(f, "3x4")
    assert T2.shape == (4,4)


def test_matrix_4x4_roundtrip():
    T = torch.eye(4)
    f = transform_to_matrix_features(T, "4x4")
    T2 = matrix_features_to_transform(f, "4x4")
    assert T2.shape == (4,4)


def test_rot6d_trans_outputs_valid_rotation():
    T = torch.eye(4)
    f = transform_to_matrix_features(T, "rot6d_trans")
    T2 = matrix_features_to_transform(f, "rot6d_trans")
    assert bool(is_valid_rotation(T2[:3,:3]).item())


def test_matrix_projection_det_positive():
    T = torch.eye(4)
    f = transform_to_matrix_features(T, "3x4")
    T2 = matrix_features_to_transform(f, "3x4")
    assert torch.det(T2[:3,:3]) > 0
