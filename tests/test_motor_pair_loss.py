import torch
from motordock.losses.motor_pair_loss import masked_pair_motor_loss


def test_pair_motor_loss_zero_for_identical_transforms():
    T = torch.eye(4).view(1,1,4,4)
    loss = masked_pair_motor_loss(T, T, torch.tensor([[1]], dtype=torch.bool))
    assert abs(float(loss.item())) < 1e-6


def test_pair_motor_loss_ignores_padded_pairs():
    T = torch.eye(4).repeat(1, 2, 1, 1)
    loss = masked_pair_motor_loss(T, T, torch.tensor([[1,0]], dtype=torch.bool))
    assert abs(float(loss.item())) < 1e-6


def test_pair_motor_loss_zero_when_no_perturbed_pairs():
    T = torch.eye(4).view(1,1,4,4)
    loss = masked_pair_motor_loss(T, T, torch.tensor([[1]], dtype=torch.bool), torch.tensor([[0]], dtype=torch.bool))
    assert abs(float(loss.item())) < 1e-8


def test_pair_motor_loss_has_gradient():
    T = torch.eye(4).view(1,1,4,4).clone().requires_grad_(True)
    T2 = torch.eye(4).view(1,1,4,4)
    loss = masked_pair_motor_loss(T, T2, torch.tensor([[1]], dtype=torch.bool))
    loss.backward()
    assert T.grad is not None
