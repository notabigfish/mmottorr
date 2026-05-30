import torch

from motordock.geometry.se3 import se3_exp_map
from motordock.geometry.pga_motor import (
    se3_to_motor,
    sandwich_points,
    normalize_motor,
    geometric_product,
    reverse,
)


def test_identity_motor_leaves_points_unchanged():
    M = torch.zeros(16)
    M[0] = 1.0
    pts = torch.randn(10, 3)
    out = sandwich_points(M, pts)
    assert torch.allclose(out, pts, atol=1e-5, rtol=1e-5)


def test_translation_maps_origin():
    T = torch.eye(4)
    T[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    M = se3_to_motor(T)
    origin = torch.zeros(1, 3)
    out = sandwich_points(M, origin)
    assert torch.allclose(out, torch.tensor([[1.0, 2.0, 3.0]]), atol=1e-4, rtol=1e-4)


def test_rotation_z90_maps_x_to_y():
    T = torch.eye(4)
    T[:3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    M = se3_to_motor(T)
    pt = torch.tensor([[1.0, 0.0, 0.0]])
    out = sandwich_points(M, pt)
    assert torch.allclose(out, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-4, rtol=1e-4)


def test_random_se3_matches_matrix_action():
    torch.manual_seed(0)
    B, N = 32, 7
    xi = torch.randn(B, 6) * 0.3
    T = se3_exp_map(xi)
    M = se3_to_motor(T)
    xyz = torch.randn(B, N, 3)

    xyz_pga = sandwich_points(M, xyz)
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    xyz_mat = xyz @ R.transpose(-1, -2) + t[:, None, :]
    assert torch.allclose(xyz_pga, xyz_mat, atol=1e-4, rtol=1e-4)


def test_sign_invariance():
    T = se3_exp_map(torch.tensor([[0.2, -0.1, 0.05, 1.0, -0.5, 0.2]])).squeeze(0)
    M = se3_to_motor(T)
    pts = torch.randn(20, 3)
    a = sandwich_points(M, pts)
    b = sandwich_points(-M, pts)
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_unit_normalization():
    T = se3_exp_map(torch.tensor([[0.1, 0.2, -0.1, 0.5, 0.1, -0.2]])).squeeze(0)
    M = se3_to_motor(T)
    M_unit = normalize_motor(2.7 * M)
    G = geometric_product(M_unit, reverse(M_unit))
    assert torch.allclose(G[..., 0], torch.ones_like(G[..., 0]), atol=1e-4, rtol=1e-4)
    assert torch.allclose(G[..., 1:], torch.zeros_like(G[..., 1:]), atol=1e-4, rtol=1e-4)


def test_autograd_through_motor_action():
    xi = torch.randn(1, 6, requires_grad=True)
    T = se3_exp_map(xi)
    M = se3_to_motor(T)
    pts = torch.randn(1, 5, 3)
    out = sandwich_points(M, pts).sum()
    out.backward()
    assert xi.grad is not None
    assert torch.isfinite(xi.grad).all()
