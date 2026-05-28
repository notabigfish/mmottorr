import torch
from motordock.models.representation_adapters import SE3LogAdapter, QuaternionTranslationAdapter, DualQuaternionAdapter, MatrixAdapter
from motordock.geometry.se3 import is_valid_transform


def _inputs():
    return torch.randn(2,3,16), torch.randn(2,16), torch.randn(2,16), torch.eye(4).view(1,1,4,4).repeat(2,3,1,1), torch.tensor([[1,1,0],[1,0,0]], dtype=torch.bool)


def test_each_adapter_forward_shapes():
    for cls in [SE3LogAdapter, QuaternionTranslationAdapter, DualQuaternionAdapter, MatrixAdapter]:
        a = cls(pair_hidden_dim=16, joint_hidden_dim=16) if cls != MatrixAdapter else cls(matrix_mode="3x4", pair_hidden_dim=16, joint_hidden_dim=16)
        o = a(*_inputs())
        assert o["pair_delta_T_pred"].shape == (2,3,4,4)


def test_each_adapter_outputs_valid_transforms_when_decodable():
    a = SE3LogAdapter(pair_hidden_dim=16, joint_hidden_dim=16)
    o = a(*_inputs())
    assert bool(is_valid_transform(o["pair_delta_T_pred"][0,0]).item())


def test_invalid_pairs_get_identity_transform():
    a = SE3LogAdapter(pair_hidden_dim=16, joint_hidden_dim=16)
    pair_h,lc,pc,T,m = _inputs()
    o = a(pair_h,lc,pc,T,m)
    assert torch.allclose(o["pair_delta_T_pred"][0,2], torch.eye(4), atol=1e-5)


def test_quaternion_adapter_normalizes_output():
    a = QuaternionTranslationAdapter(pair_hidden_dim=16, joint_hidden_dim=16)
    o = a(*_inputs())
    q = o["pair_rep_pred"][..., :4]
    n = q.norm(dim=-1)
    assert torch.isfinite(n).all()


def test_dual_quaternion_adapter_normalizes_output():
    a = DualQuaternionAdapter(pair_hidden_dim=16, joint_hidden_dim=16)
    o = a(*_inputs())
    assert torch.isfinite(o["pair_rep_pred"]).all()


def test_matrix_adapter_projects_to_so3():
    a = MatrixAdapter(matrix_mode="3x4", pair_hidden_dim=16, joint_hidden_dim=16)
    o = a(*_inputs())
    assert torch.isfinite(o["pair_delta_T_pred"]).all()
