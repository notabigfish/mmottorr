import torch

from motordock.geometry.frames import (
    weighted_pca_frame,
    make_frame_matrix,
    compute_pair_transform_from_frames,
)
from motordock.geometry.se3 import is_valid_transform


def test_weighted_pca_frame_valid_rotation():
    torch.manual_seed(0)
    coords = torch.randn(20, 3, dtype=torch.float64)
    pocket = torch.zeros(3, dtype=torch.float64)
    fr = weighted_pca_frame(coords, pocket)
    assert fr.R is not None
    assert fr.t is not None
    I = torch.eye(3, dtype=torch.float64)
    assert torch.allclose(fr.R.T @ fr.R, I, atol=1e-5)
    assert abs(torch.det(fr.R).item() - 1.0) < 1e-5


def test_weighted_pca_frame_too_few_points():
    coords = torch.randn(4, 3)
    pocket = torch.zeros(3)
    fr = weighted_pca_frame(coords, pocket, min_points=8)
    assert fr.stable is False
    assert fr.reason == "too_few_points"


def test_weighted_pca_frame_detects_collinear_points():
    x = torch.linspace(-5, 5, 20)
    coords = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=-1)
    pocket = torch.zeros(3)
    fr = weighted_pca_frame(coords, pocket, eig_ratio_threshold=0.2)
    assert fr.stable is False
    assert fr.reason.startswith("eig_ratio_below_threshold")


def test_make_frame_matrix_valid_transform():
    R = torch.eye(3)
    t = torch.tensor([1.0, 2.0, 3.0])
    T = make_frame_matrix(R, t)
    assert T.shape == (4, 4)
    assert is_valid_transform(T).item()


def test_compute_pair_transform_from_frames():
    Ta = torch.eye(4)
    Tb = torch.eye(4)
    Tb[:3, 3] = torch.tensor([1.0, 0.0, 0.0])
    Tab = compute_pair_transform_from_frames(Ta, Tb)
    assert torch.allclose(Tab, Tb)
