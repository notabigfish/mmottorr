from __future__ import annotations

from motordock.data.motordock_dataset import MotorDockSE3Dataset
from .representation_pair_featurizer import featurize_candidate_pairs_with_representation
import torch

class RepresentationAblationDataset(MotorDockSE3Dataset):
    def __init__(self, *args, representation: str = "se3_log", matrix_mode: str = "3x4", ablation_mode: str = "normal", shuffle_pair_features: bool = False, random_motor_seed: int = 0, **kwargs):
        self.representation = representation
        self.matrix_mode = matrix_mode
        self.ablation_mode = ablation_mode
        self.shuffle_pair_features = shuffle_pair_features
        self.random_motor_seed = random_motor_seed
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        s = super().__getitem__(idx)
        C = s["pair_features"].shape[0]

        pairs = []
        for i in range(C):
            pairs.append({
                "pair_id": s["pair_ids"][i] if i < len(s["pair_ids"]) else f"pair_{i}",
                "pair_type": s["pair_types"][i] if i < len(s["pair_types"]) else "unknown",
                "unit_a_type": "unknown",
                "unit_b_type": "unknown",
                "unit_a_chain": "",
                "unit_b_chain": "",
                "unit_a_num_residues": 0,
                "unit_b_num_residues": 0,
                "frame_a_stable": bool(s["pair_valid"][i].item()),
                "frame_b_stable": bool(s["pair_valid"][i].item()),
                "has_native_transform": bool(s["pair_valid"][i].item()),
                "T_ab_recomputed": s["pair_T_native"][i],
            })

        rep_name = self.representation
        if self.ablation_mode in {"random_motor", "shuffled_pairs", "no_pair_context"}:
            rep_name = "se3_log" if self.ablation_mode != "random_motor" else "random_motor"

        feats, ids, types = featurize_candidate_pairs_with_representation(
            pairs,
            representation=rep_name,
            matrix_mode=self.matrix_mode,
            random_seed=self.random_motor_seed + idx,
        )

        if self.ablation_mode == "shuffled_pairs" or self.shuffle_pair_features:
            perm = torch.randperm(feats.shape[0])
            feats = feats[perm]
            ids = [ids[i] for i in perm.tolist()]
            types = [types[i] for i in perm.tolist()]

        if self.ablation_mode == "no_pair_context":
            s["pair_mask"] = s["pair_mask"] & False

        s["pair_features"] = feats
        s["pair_ids"] = ids
        s["pair_types"] = types
        s["representation"] = self.representation
        return s
