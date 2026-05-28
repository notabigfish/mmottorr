import torch

from motordock.geometry.se3 import (
    skew,
    unskew,
    make_transform,
    inverse_transform,
    compose_transform,
    so3_exp_map,
    so3_log_map,
    se3_exp_map,
    se3_log_map,
    se3_geodesic_loss,
    project_to_so3,
)


def _atol(dtype):
    return 1e-5 if dtype == torch.float64 else 1e-4


def test_skew_unskew_roundtrip():
    w = torch.randn(5, 3, dtype=torch.float64)
    W = skew(w)
    w2 = unskew(W)
    assert torch.allclose(w, w2, atol=1e-8)


def test_make_inverse_identity():
    xi = torch.randn(3, 6, dtype=torch.float64) * 0.1
    T = se3_exp_map(xi)
    I = compose_transform(inverse_transform(T), T)
    eye = torch.eye(4, dtype=torch.float64).expand_as(I)
    assert torch.allclose(I, eye, atol=1e-6)


def test_exp_log_roundtrip_small_rotation():
    xi = torch.tensor([[1e-6, -2e-6, 3e-6, 0.1, -0.2, 0.3]], dtype=torch.float64)
    T = se3_exp_map(xi)
    xi2 = se3_log_map(T)
    assert torch.allclose(xi, xi2, atol=1e-5)


def test_exp_log_roundtrip_random_batch():
    xi = torch.randn(16, 6, dtype=torch.float32) * 0.2
    T = se3_exp_map(xi)
    xi2 = se3_log_map(T)
    assert torch.allclose(xi, xi2, atol=1e-3)


def test_geodesic_loss_zero_for_identical_transform():
    xi = torch.randn(4, 6, dtype=torch.float32) * 0.1
    T = se3_exp_map(xi)
    loss = se3_geodesic_loss(T, T)
    assert torch.isclose(loss, torch.tensor(0.0, dtype=loss.dtype), atol=1e-6)


def test_project_to_so3_returns_det_positive():
    R = torch.eye(3).repeat(4, 1, 1)
    R[0, 0, 0] = -1.0
    Rp = project_to_so3(R)
    det = torch.det(Rp)
    assert torch.all(det > 0)


def test_se3_preserves_dtype_and_device():
    for dtype in (torch.float32, torch.float64):
        xi = torch.randn(2, 6, dtype=dtype)
        T = se3_exp_map(xi)
        assert T.dtype == dtype
        xi2 = se3_log_map(T)
        assert xi2.dtype == dtype
        assert torch.allclose(xi, xi2, atol=_atol(dtype) * 10)
