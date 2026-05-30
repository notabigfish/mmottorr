from __future__ import annotations

import torch
from .collate import baseline_collate_fn


def motordock_se3_collate_fn(batch: list[dict]) -> dict:
    out = baseline_collate_fn(batch)
    B = len(batch)
    Cmax = max(x["pair_features"].shape[0] for x in batch)
    Fp = batch[0]["pair_features"].shape[1]

    pair_features = torch.zeros((B, Cmax, Fp), dtype=torch.float32)
    pair_mask = torch.zeros((B, Cmax), dtype=torch.bool)
    pair_valid = torch.zeros((B, Cmax), dtype=torch.bool)
    pair_was_perturbed = torch.zeros((B, Cmax), dtype=torch.bool)
    pair_T_input = torch.eye(4, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(B, Cmax, 1, 1)
    pair_T_native = torch.eye(4, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(B, Cmax, 1, 1)
    pair_T_target_residual = torch.eye(4, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(B, Cmax, 1, 1)

    pair_ids = []
    pair_types = []
    for i, s in enumerate(batch):
        c = s["pair_features"].shape[0]
        pair_features[i, :c] = s["pair_features"]
        pair_mask[i, :c] = s["pair_mask"]
        pair_valid[i, :c] = s["pair_valid"]
        pair_was_perturbed[i, :c] = s["pair_was_perturbed"]
        pair_T_input[i, :c] = s["pair_T_input"]
        pair_T_native[i, :c] = s["pair_T_native"]
        pair_T_target_residual[i, :c] = s["pair_T_target_residual"]
        pair_ids.append(list(s["pair_ids"]))
        pair_types.append(list(s["pair_types"]))

    out.update({
        "pair_features": pair_features,
        "pair_mask": pair_mask,
        "pair_T_input": pair_T_input,
        "pair_T_native": pair_T_native,
        "pair_T_target_residual": pair_T_target_residual,
        "pair_valid": pair_valid,
        "pair_was_perturbed": pair_was_perturbed,
        "pair_ids": pair_ids,
        "pair_types": pair_types,
    })
    return out
