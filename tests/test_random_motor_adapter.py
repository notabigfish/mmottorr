import torch
import torch.nn as nn

from motordock.models.representation_adapters import RandomMotorAdapter, SE3LogAdapter


def _inputs(B=2, C=3, H=16):
    pair_h = torch.randn(B, C, H)
    l_ctx = torch.randn(B, H)
    p_ctx = torch.randn(B, H)
    T = torch.eye(4).view(1, 1, 4, 4).repeat(B, C, 1, 1)
    m = torch.ones((B, C), dtype=torch.bool) 
    return pair_h, l_ctx, p_ctx, T, m


def test_no_trainable_parameters():
    a = RandomMotorAdapter()
    n = sum(p.numel() for p in a.parameters() if p.requires_grad)
    assert n == 0


def test_deterministic_per_complex():
    a = RandomMotorAdapter(seed=7, mode="fixed_per_complex")
    pair_h, l_ctx, p_ctx, T, m = _inputs(B=3, C=2)
    batch = {"complex_id": ["A", "B", "A"]}

    o1 = a(pair_h, l_ctx, p_ctx, T, m, batch=batch)
    o2 = a(pair_h, l_ctx, p_ctx, T, m, batch=batch)

    assert torch.allclose(o1["pair_xi_pred"], o2["pair_xi_pred"])
    assert torch.allclose(o1["pair_xi_pred"][0], o1["pair_xi_pred"][2])
    assert not torch.allclose(o1["pair_xi_pred"][0], o1["pair_xi_pred"][1])


def test_valid_se3():
    a = RandomMotorAdapter(seed=1)
    pair_h, l_ctx, p_ctx, T, m = _inputs()
    o = a(pair_h, l_ctx, p_ctx, T, m, batch={"complex_id": ["X", "Y"]})
    D = o["pair_delta_T_pred"]
    R = D[..., :3, :3]
    b = D[..., 3, :]

    I = torch.eye(3).to(R)
    RtR = R.transpose(-1, -2) @ R
    det = torch.det(R)

    assert torch.allclose(b, torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=b.dtype, device=b.device).view(1, 1, 4).expand_as(b), atol=1e-5)
    assert torch.allclose(RtR, I.view(1, 1, 3, 3).expand_as(RtR), atol=1e-4)
    assert torch.allclose(det, torch.ones_like(det), atol=1e-4)


def test_no_gradient_outputs():
    a = RandomMotorAdapter()
    pair_h, l_ctx, p_ctx, T, m = _inputs()
    o = a(pair_h, l_ctx, p_ctx, T, m)
    assert o["pair_xi_pred"].requires_grad is False
    assert o["pair_delta_T_pred"].requires_grad is False


def test_fresh_each_call_changes_and_reproducible_first_call():
    a1 = RandomMotorAdapter(seed=9, mode="fresh_each_call")
    pair_h, l_ctx, p_ctx, T, m = _inputs()
    o1 = a1(pair_h, l_ctx, p_ctx, T, m)
    o2 = a1(pair_h, l_ctx, p_ctx, T, m)
    assert not torch.allclose(o1["pair_xi_pred"], o2["pair_xi_pred"])

    a2 = RandomMotorAdapter(seed=9, mode="fresh_each_call")
    o3 = a2(pair_h, l_ctx, p_ctx, T, m)
    assert torch.allclose(o1["pair_xi_pred"], o3["pair_xi_pred"])


def test_not_equivalent_to_se3log_adapter():
    a = RandomMotorAdapter()
    assert not isinstance(a, SE3LogAdapter)
    assert not any(isinstance(m, nn.Linear) for m in a.modules())
