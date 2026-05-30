from __future__ import annotations

import torch


class OneStepPoseSampler:
    def __init__(self, model, num_samples: int):
        self.model = model
        self.num_samples = int(num_samples)

    @torch.no_grad()
    def sample(self, batch: dict) -> dict:
        B = batch["ligand_coords_start"].shape[0]
        coords = []
        confs = []
        for _ in range(self.num_samples):
            out = self.model(batch)
            coords.append(out["ligand_coords_pred"])
            confs.append(out["confidence_logit"])
        coords = torch.stack(coords, dim=1)
        confs = torch.stack(confs, dim=1)
        ranked = torch.argsort(confs, dim=1, descending=True)
        return {
            "coords": coords,
            "confidence_logit": confs,
            "ranked_indices": ranked,
        }


def sample_indices(num_samples: int):
    return list(range(num_samples))
