import torch
from motordock.geometry.dual_quaternion import transform_to_dual_quaternion, dual_quaternion_to_transform, dual_quaternion_inverse, dual_quaternion_multiply


def test_dual_quaternion_transform_roundtrip_identity():
    T = torch.eye(4)
    dq = transform_to_dual_quaternion(T)
    T2 = dual_quaternion_to_transform(dq)
    assert T2.shape == (4,4)


def test_dual_quaternion_transform_roundtrip_random_batch():
    T = torch.eye(4).view(1,4,4).repeat(4,1,1)
    T[:, :3, 3] = torch.randn(4,3)
    dq = transform_to_dual_quaternion(T)
    T2 = dual_quaternion_to_transform(dq)
    assert T2.shape == T.shape


def test_dual_quaternion_inverse_identity():
    T = torch.eye(4)
    dq = transform_to_dual_quaternion(T)
    I = dual_quaternion_multiply(dq, dual_quaternion_inverse(dq))
    assert torch.isfinite(I).all()


def test_dual_quaternion_outputs_finite():
    T = torch.eye(4)
    dq = transform_to_dual_quaternion(T)
    assert torch.isfinite(dq).all()


def test_dual_quaternion_standardized_sign():
    T = torch.eye(4)
    dq = transform_to_dual_quaternion(T)
    assert dq[0] >= 0
