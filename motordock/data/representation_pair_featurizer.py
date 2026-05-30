from __future__ import annotations

import hashlib
import torch
from motordock.geometry.representation_conversions import transform_to_representation, representation_dim
from motordock.data.pair_featurizer import _PAIR_TYPES, _UNIT_TYPES


def _one_hot(v: str, vocab: list[str]):
    return [1.0 if v == x else 0.0 for x in vocab]


def _meta_features(pair: dict) -> torch.Tensor:
    pt = str(pair.get("pair_type", "unknown"))
    if pt not in _PAIR_TYPES:
        pt = "unknown"
    ua = str(pair.get("unit_a_type", "unknown"))
    ub = str(pair.get("unit_b_type", "unknown"))
    if ua not in _UNIT_TYPES: ua = "unknown"
    if ub not in _UNIT_TYPES: ub = "unknown"

    same_chain = 1.0 if str(pair.get("unit_a_chain", "")) == str(pair.get("unit_b_chain", "")) else 0.0
    na = float(pair.get("unit_a_num_residues", 0) or 0)
    nb = float(pair.get("unit_b_num_residues", 0) or 0)

    feats = []
    feats.extend(_one_hot(pt, _PAIR_TYPES))
    feats.extend(_one_hot(ua, _UNIT_TYPES))
    feats.extend(_one_hot(ub, _UNIT_TYPES))
    feats.extend([same_chain, 1.0 - same_chain])
    la = torch.log1p(torch.tensor(na)).item()
    lb = torch.log1p(torch.tensor(nb)).item()
    feats.extend([la, lb, abs(la - lb)])
    feats.extend([
        float(bool(pair.get("frame_a_stable", False))),
        float(bool(pair.get("frame_b_stable", False))),
        float(bool(pair.get("has_native_transform", False))),
    ])
    return torch.tensor(feats, dtype=torch.float32)


def _centroid_bias(T_ab: torch.Tensor) -> torch.Tensor:
    t = T_ab[:3, 3]
    # no full orientation; coarse geometry only
    return torch.tensor([t.norm().item(), abs(t[0].item()), abs(t[1].item()), abs(t[2].item())], dtype=torch.float32)


def featurize_candidate_pair_with_representation(
    pair: dict,
    T_ab: torch.Tensor,
    representation: str = "se3_log",
    matrix_mode: str = "3x4",
    random_seed: int | None = None,
) -> torch.Tensor:
    meta = _meta_features(pair)
    if not torch.is_tensor(T_ab):
        T_ab = torch.tensor(T_ab, dtype=torch.float32)
    T_ab = T_ab.float()

    if representation == "centroid_bias":
        rep = _centroid_bias(T_ab)
    elif representation == "random_motor":
        dim = representation_dim("random_motor", matrix_mode)
        if random_seed is None:
            random_seed = 0
        g = torch.Generator().manual_seed(int(random_seed))
        rep = torch.randn(dim, generator=g, dtype=torch.float32)
    elif representation == "pga_feature":
        # passive motor-like feature baseline only
        se3 = transform_to_representation(T_ab.unsqueeze(0), "dual_quaternion").squeeze(0)
        rep = se3
    elif representation == "matrix":
        rep = transform_to_representation(T_ab.unsqueeze(0), "matrix").squeeze(0)
    elif representation == "shuffled_pairs":
        rep = transform_to_representation(T_ab.unsqueeze(0), "se3_log").squeeze(0)
    elif representation == "no_pair_context":
        rep = transform_to_representation(T_ab.unsqueeze(0), "se3_log").squeeze(0)
    else:
        rep = transform_to_representation(T_ab.unsqueeze(0), representation).squeeze(0)

    return torch.cat([rep.float(), meta], dim=0)


def representation_pair_feature_dim(representation: str, matrix_mode: str = "3x4") -> int:
    meta_dim = 5 + 4 + 4 + 2 + 3 + 3
    if representation == "centroid_bias":
        return 4 + meta_dim
    return representation_dim(representation if representation != "shuffled_pairs" else "se3_log", matrix_mode) + meta_dim


def featurize_candidate_pairs_with_representation(candidate_pairs: list[dict], representation: str, matrix_mode: str = "3x4", random_seed: int | None = None) -> tuple[torch.Tensor, list[str], list[str]]:
    feats, ids, types = [], [], []
    for p in candidate_pairs:
        T = p.get("T_ab_recomputed", p.get("T_ab_native", torch.eye(4)))
        if not torch.is_tensor(T):
            T = torch.tensor(T, dtype=torch.float32)
        pid = str(p.get("pair_id", "unknown"))
        seed = random_seed
        if representation == "random_motor":
            h = int(hashlib.md5(pid.encode()).hexdigest()[:8], 16)
            seed = (0 if random_seed is None else random_seed) + h
        feats.append(featurize_candidate_pair_with_representation(p, T, representation=representation, matrix_mode=matrix_mode, random_seed=seed))
        ids.append(pid)
        types.append(str(p.get("pair_type", "unknown")))
    if not feats:
        feats = [torch.zeros(representation_pair_feature_dim(representation, matrix_mode), dtype=torch.float32)]
        ids = ["identity_fallback"]
        types = ["unknown"]
    return torch.stack(feats, dim=0), ids, types
