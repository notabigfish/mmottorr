import torch

from motordock.models.pga_adapter import PGAFeatureAdapter, PGASandwichAdapter
from motordock.models.representation_adapters import CentroidBiasAdapter


def _pair_inputs(B=2, C=3, H=16):
    T = torch.eye(4).view(1, 1, 4, 4).repeat(B, C, 1, 1)
    T[:, :, :3, 3] = torch.randn(B, C, 3)
    pair_h = torch.randn(B, C, H)
    return {"pair_T_initial": T, "pair_h": pair_h}


def test_pga_feature_adapter_not_centroid_bias():
    adapter = PGAFeatureAdapter(in_dim=16, hidden_dim=16, out_dim=16)
    assert not isinstance(adapter, CentroidBiasAdapter)


def test_pga_feature_forward_keys():
    adapter = PGAFeatureAdapter(in_dim=16, hidden_dim=16, out_dim=16)
    out = adapter(_pair_inputs())
    assert "pga_motor" in out
    assert "pga_motor_features" in out
    assert "pga_context" in out


def test_pga_sandwich_forward_keys():
    adapter = PGASandwichAdapter(in_dim=16, hidden_dim=16, out_dim=16)
    out = adapter(_pair_inputs())
    assert "pga_motor" in out
    assert "pga_transformed_points" in out
    assert "pga_action_features" in out
    assert "pga_context" in out


def test_pga_sandwich_changes_when_transform_changes():
    adapter = PGASandwichAdapter(in_dim=16, hidden_dim=16, out_dim=16)
    inp1 = _pair_inputs(B=1, C=1)
    inp2 = {k: v.clone() for k, v in inp1.items()}
    inp2["pair_T_initial"][0, 0, :3, 3] += torch.tensor([0.5, -0.2, 0.3])
    o1 = adapter(inp1)["pga_context"]
    o2 = adapter(inp2)["pga_context"]
    assert not torch.allclose(o1, o2)


def test_sandwich_and_feature_outputs_differ():
    feature = PGAFeatureAdapter(in_dim=16, hidden_dim=16, out_dim=16)
    sandwich = PGASandwichAdapter(in_dim=16, hidden_dim=16, out_dim=16)
    inp = _pair_inputs(B=1, C=1)
    of = feature(inp)["pga_context"]
    os = sandwich(inp)["pga_context"]
    assert not torch.allclose(of, os)


def test_pga_sandwich_calls_sandwich_points(monkeypatch):
    from motordock.models import pga_adapter as pa

    called = {"flag": False}
    real = pa.sandwich_points

    def spy(*args, **kwargs):
        called["flag"] = True
        return real(*args, **kwargs)

    monkeypatch.setattr(pa, "sandwich_points", spy)
    adapter = PGASandwichAdapter(in_dim=16, hidden_dim=16, out_dim=16)
    _ = adapter(_pair_inputs(B=1, C=1))
    assert called["flag"]
