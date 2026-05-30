from __future__ import annotations

import math
import torch
from motordock.geometry.se3 import se3_log_map

_PAIR_TYPES = ["single", "domain_pair", "chain_pair", "mixed_pair", "unknown"]
_UNIT_TYPES = ["pfam_domain", "linker", "chain", "unknown"]


def _one_hot(v: str, vocab: list[str]) -> list[float]:
    return [1.0 if v == x else 0.0 for x in vocab]


def _safe_pair_type(pair: dict) -> str:
    t = str(pair.get("pair_type", "unknown"))
    return t if t in _PAIR_TYPES else "unknown"


def _safe_unit_type(pair: dict, key: str) -> str:
    t = str(pair.get(key, "unknown"))
    return t if t in _UNIT_TYPES else "unknown"


def featurize_candidate_pair(pair: dict, T_ab: torch.Tensor) -> torch.Tensor:
    if T_ab.shape != (4, 4):
        T_ab = torch.eye(4, dtype=torch.float32)
    T_ab = T_ab.float()
    xi = se3_log_map(T_ab)
    rot = xi[:3]
    trans = xi[3:]
    pair_type = _safe_pair_type(pair)
    ua_type = _safe_unit_type(pair, "unit_a_type")
    ub_type = _safe_unit_type(pair, "unit_b_type")
    ua_chain = str(pair.get("unit_a_chain", ""))
    ub_chain = str(pair.get("unit_b_chain", ""))
    same_chain = 1.0 if ua_chain == ub_chain and ua_chain != "" else 0.0

    na = float(pair.get("unit_a_num_residues", 0) or 0)
    nb = float(pair.get("unit_b_num_residues", 0) or 0)
    la = math.log1p(max(na, 0.0))
    lb = math.log1p(max(nb, 0.0))

    feats = []
    feats.extend(xi.tolist())
    feats.append(float(torch.linalg.norm(trans).item()))
    feats.append(float(torch.linalg.norm(rot).item()))
    feats.extend(_one_hot(pair_type, _PAIR_TYPES))
    feats.extend(_one_hot(ua_type, _UNIT_TYPES))
    feats.extend(_one_hot(ub_type, _UNIT_TYPES))
    feats.extend([same_chain, 1.0 - same_chain])
    feats.extend([la, lb, abs(la - lb)])
    feats.extend([
        float(bool(pair.get("frame_a_stable", False))),
        float(bool(pair.get("frame_b_stable", False))),
        float(bool(pair.get("has_native_transform", False))),
    ])
    return torch.tensor(feats, dtype=torch.float32)


def pair_feature_dim() -> int:
    # 6 + 1 + 1 + 5 + 4 + 4 + 2 + 3 + 3
    return 29


def featurize_candidate_pairs(candidate_pairs: list[dict]) -> tuple[torch.Tensor, list[str], list[str]]:
    feats = []
    pair_ids: list[str] = []
    pair_types: list[str] = []
    for p in candidate_pairs:
        T = p.get("T_ab_recomputed", p.get("T_ab_native", torch.eye(4)))
        if not torch.is_tensor(T):
            T = torch.tensor(T, dtype=torch.float32)
        if T.shape != (4, 4):
            T = torch.eye(4, dtype=torch.float32)
        feats.append(featurize_candidate_pair(p, T))
        pair_ids.append(str(p.get("pair_id", "unknown_pair")))
        pair_types.append(_safe_pair_type(p))
    if not feats:
        feats = [torch.zeros(pair_feature_dim(), dtype=torch.float32)]
        pair_ids = ["identity_fallback"]
        pair_types = ["unknown"]
    out = torch.stack(feats, dim=0)
    return out, pair_ids, pair_types
