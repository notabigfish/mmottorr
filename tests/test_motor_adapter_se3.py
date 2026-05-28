import torch
from motordock.models.motor_adapter_se3 import MotorAdapterSE3
from motordock.geometry.se3 import is_valid_transform


def test_motor_adapter_shapes():
    m = MotorAdapterSE3(16, 16)
    out = m(torch.randn(2, 3, 16), torch.randn(2, 16), torch.randn(2, 16), torch.eye(4).view(1, 1, 4, 4).repeat(2, 3, 1, 1), torch.tensor([[1,1,0],[1,0,0]], dtype=torch.bool))
    assert out["pair_xi_pred"].shape == (2, 3, 6)


def test_motor_adapter_outputs_valid_transforms():
    m = MotorAdapterSE3(16, 16)
    out = m(torch.randn(1, 2, 16), torch.randn(1, 16), torch.randn(1, 16), torch.eye(4).view(1,1,4,4).repeat(1,2,1,1), torch.tensor([[1,1]], dtype=torch.bool))
    assert bool(is_valid_transform(out["pair_delta_T_pred"][0,0]).item())


def test_motor_adapter_invalid_pairs_are_identity():
    m = MotorAdapterSE3(16, 16)
    out = m(torch.randn(1, 2, 16), torch.randn(1, 16), torch.randn(1, 16), torch.eye(4).view(1,1,4,4).repeat(1,2,1,1), torch.tensor([[1,0]], dtype=torch.bool))
    assert torch.allclose(out["pair_delta_T_pred"][0,1], torch.eye(4))


def test_motor_adapter_outputs_finite_values():
    m = MotorAdapterSE3(16, 16)
    out = m(torch.randn(2, 3, 16), torch.randn(2, 16), torch.randn(2, 16), torch.eye(4).view(1,1,4,4).repeat(2,3,1,1), torch.ones(2,3, dtype=torch.bool))
    assert torch.isfinite(out["pair_xi_pred"]).all()
