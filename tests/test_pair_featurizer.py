import torch
from motordock.data.pair_featurizer import pair_feature_dim, featurize_candidate_pair, featurize_candidate_pairs


def test_pair_feature_dim_fixed():
    assert pair_feature_dim() == 29


def test_pair_features_are_finite():
    p = {"pair_type": "domain_pair", "unit_a_type": "pfam_domain", "unit_b_type": "chain"}
    f = featurize_candidate_pair(p, torch.eye(4))
    assert torch.isfinite(f).all()


def test_pair_features_include_se3_log_values():
    T = torch.eye(4)
    T[:3, 3] = torch.tensor([1.0, 0.0, 0.0])
    p = {"pair_type": "single", "unit_a_type": "unknown", "unit_b_type": "unknown"}
    f = featurize_candidate_pair(p, T)
    assert abs(float(f[3].item())) > 0.0


def test_pair_featurizer_handles_missing_optional_keys():
    feats, ids, types = featurize_candidate_pairs([{}])
    assert feats.shape[1] == pair_feature_dim()
    assert len(ids) == 1 and len(types) == 1
