from __future__ import annotations

import torch

AA = "ARNDCQEGHILKMFPSTWYV"
AA_IDX = {a: i for i, a in enumerate(AA)}


def featurize_protein_sequence(sequence: str, pocket_mask: torch.Tensor, domain_mask: torch.Tensor) -> torch.Tensor:
    n = len(sequence)
    aa = torch.zeros((n, 21), dtype=torch.float32)
    for i, s in enumerate(sequence):
        idx = AA_IDX.get(s, 20)
        aa[i, idx] = 1.0

    pocket = pocket_mask.float().view(n, 1)
    d = domain_mask.float()
    d = d / d.clamp_min(1.0).max()
    d = d.view(n, 1)

    if n > 1:
        rel = torch.linspace(0.0, 1.0, n, dtype=torch.float32).view(n, 1)
    else:
        rel = torch.zeros((n, 1), dtype=torch.float32)

    return torch.cat([aa, pocket, d, rel], dim=-1)
