import torch

from motordock.geometry.se3 import se3_exp_map
from motordock.geometry.pga_motor import se3_to_motor
from motordock.losses.representation_loss import pga_motor_loss, masked_representation_motor_loss


def test_pga_motor_loss_zero_identical():
    T = se3_exp_map(torch.tensor([[0.1, 0.2, -0.1, 0.5, -0.3, 0.2]])).squeeze(0)
    M = se3_to_motor(T)
    loss = pga_motor_loss(M.unsqueeze(0), M.unsqueeze(0), T.unsqueeze(0), T.unsqueeze(0))
    assert float(loss) < 1e-6


def test_pga_motor_loss_positive_different():
    T1 = se3_exp_map(torch.tensor([[0.1, 0.2, -0.1, 0.5, -0.3, 0.2]])).squeeze(0)
    T2 = se3_exp_map(torch.tensor([[0.2, -0.1, 0.1, -0.5, 0.2, 0.4]])).squeeze(0)
    M1 = se3_to_motor(T1)
    M2 = se3_to_motor(T2)
    loss = pga_motor_loss(M1.unsqueeze(0), M2.unsqueeze(0), T1.unsqueeze(0), T2.unsqueeze(0))
    assert float(loss) > 1e-6


def test_pga_motor_loss_sign_invariant():
    T = se3_exp_map(torch.tensor([[0.1, -0.05, 0.2, 0.1, 0.2, 0.3]])).squeeze(0)
    M = se3_to_motor(T)
    loss = pga_motor_loss(M.unsqueeze(0), (-M).unsqueeze(0), T.unsqueeze(0), T.unsqueeze(0))
    assert float(loss) < 1e-6


def test_representation_dispatch_nonzero_for_pga_feature():
    B, C = 1, 2
    T_pred = torch.eye(4).view(1, 1, 4, 4).repeat(B, C, 1, 1)
    T_tgt = T_pred.clone()
    T_tgt[0, 0, :3, 3] = torch.tensor([1.0, 0.0, 0.0])

    M_pred = se3_to_motor(T_pred.reshape(-1, 4, 4)).reshape(B, C, 16)

    outputs = {
        "pair_delta_T_pred": torch.eye(4).view(1, 1, 4, 4).repeat(B, C, 1, 1),
        "pga_motor": M_pred,
        "pair_T_corrected": T_pred,
    }
    batch = {
        "pair_T_target_residual": T_tgt,
        "pair_mask": torch.tensor([[True, True]]),
        "pair_valid": torch.tensor([[True, True]]),
    }
    loss = masked_representation_motor_loss(outputs, batch, "pga_feature", batch["pair_mask"])
    assert float(loss) > 1e-6
