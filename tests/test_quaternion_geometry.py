import torch
from motordock.geometry.quaternion import normalize_quaternion, quaternion_to_matrix, matrix_to_quaternion, transform_to_quat_trans, quat_trans_to_transform


def test_quaternion_normalization():
    q = torch.randn(5,4)
    qn = normalize_quaternion(q)
    assert torch.allclose(qn.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_quaternion_matrix_roundtrip_identity():
    q = torch.tensor([[1.0,0,0,0]])
    R = quaternion_to_matrix(q)
    q2 = matrix_to_quaternion(R)
    assert torch.allclose(q2, q, atol=1e-5)


def test_quaternion_matrix_roundtrip_random():
    q = normalize_quaternion(torch.randn(10,4))
    R = quaternion_to_matrix(q)
    q2 = matrix_to_quaternion(R)
    assert torch.isfinite(q2).all()


def test_quat_trans_transform_roundtrip():
    T = torch.eye(4).view(1,4,4).repeat(3,1,1)
    T[:, :3, 3] = torch.randn(3,3)
    qt = transform_to_quat_trans(T)
    T2 = quat_trans_to_transform(qt)
    assert T2.shape == T.shape


def test_quat_trans_outputs_finite():
    T = torch.eye(4)
    qt = transform_to_quat_trans(T)
    assert torch.isfinite(qt).all()
