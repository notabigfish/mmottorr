from __future__ import annotations

from pathlib import Path
import random
import torch
from torch.utils.data import Dataset

from .pdbbind_baseline_dataset import PDBBindBaselineDataset
from .pair_featurizer import featurize_candidate_pairs
from motordock.geometry.transforms_io import (
    load_domain_frames,
    load_candidate_pairs,
    build_unit_frame_map,
    attach_pair_transforms,
)
from motordock.geometry.se3 import inverse_transform, compose_transform
from motordock.geometry.perturbation import perturb_transform


class MotorDockSE3Dataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        output_dir: str,
        split: str | None = None,
        max_examples: int | None = None,
        require_pocket: bool = True,
        max_ligand_atoms: int = 128,
        max_protein_residues: int = 1022,
        max_candidate_pairs: int = 16,
        sanitize_ligand: bool = True,
        randomize_pose: bool = True,
        max_translation: float = 10.0,
        max_rotation_degrees: float = 180.0,
        perturb_pair_transform: bool = True,
        pair_perturb_prob: float = 0.5,
        pair_max_rotation_degrees: float = 10.0,
        pair_max_translation: float = 2.0,
        seed: int = 0,
    ):
        self.base = PDBBindBaselineDataset(
            csv_path=csv_path,
            output_dir=output_dir,
            split=split,
            max_examples=max_examples,
            require_pocket=require_pocket,
            max_ligand_atoms=max_ligand_atoms,
            max_protein_residues=max_protein_residues,
            sanitize_ligand=sanitize_ligand,
            randomize_pose=randomize_pose,
            max_translation=max_translation,
            max_rotation_degrees=max_rotation_degrees,
            seed=seed,
        )
        self.output_dir = Path(output_dir)
        self.max_candidate_pairs = max_candidate_pairs
        self.perturb_pair_transform = perturb_pair_transform
        self.pair_perturb_prob = pair_perturb_prob
        self.pair_max_rotation_degrees = pair_max_rotation_degrees
        self.pair_max_translation = pair_max_translation
        self.seed = seed

    def __len__(self):
        return len(self.base)

    def _fallback_candidate(self):
        I = torch.eye(4, dtype=torch.float32)
        p = {
            "pair_id": "identity_fallback",
            "pair_type": "unknown",
            "unit_a_type": "unknown",
            "unit_b_type": "unknown",
            "unit_a_chain": "",
            "unit_b_chain": "",
            "unit_a_num_residues": 0,
            "unit_b_num_residues": 0,
            "frame_a_stable": False,
            "frame_b_stable": False,
            "has_native_transform": False,
            "T_ab_recomputed": I,
            "T_ab_native": I,
            "pair_valid": False,
        }
        return [p]

    def _rank_pairs(self, pairs: list[dict]) -> list[dict]:
        def key(p):
            valid = 1 if bool(p.get("pair_valid", False)) else 0
            non_chain = 1 if str(p.get("pair_type", "")) != "chain_pair" else 0
            stable = 1 if bool(p.get("frame_a_stable", False) and p.get("frame_b_stable", False)) else 0
            return (-valid, -stable, -non_chain)
        return sorted(pairs, key=key)

    def __getitem__(self, idx):
        sample = self.base[idx]
        pdb_id = sample["pdb_id"]

        frames = load_domain_frames(self.output_dir, pdb_id)
        cands = load_candidate_pairs(self.output_dir, pdb_id)
        fmap = build_unit_frame_map(frames)
        cands = attach_pair_transforms(cands, fmap)

        valid_pairs = []
        for p in cands:
            T = p.get("T_ab_recomputed", None)
            if T is None:
                T = p.get("T_ab_native", None)
            if T is None:
                continue
            if not torch.is_tensor(T):
                T = torch.tensor(T, dtype=torch.float32)
            if T.shape != (4, 4):
                continue
            p2 = dict(p)
            p2["T_ab_recomputed"] = T.float()
            p2["pair_valid"] = True
            valid_pairs.append(p2)

        if not valid_pairs:
            valid_pairs = self._fallback_candidate()

        valid_pairs = self._rank_pairs(valid_pairs)[: self.max_candidate_pairs]

        pair_features, pair_ids, pair_types = featurize_candidate_pairs(valid_pairs)
        C = pair_features.shape[0]

        pair_mask = torch.ones(C, dtype=torch.bool)
        pair_valid = torch.tensor([bool(p.get("pair_valid", False)) for p in valid_pairs], dtype=torch.bool)

        pair_T_native = []
        pair_T_input = []
        pair_T_target_residual = []
        pair_was_perturbed = []

        rng = random.Random(self.seed + idx)
        for p in valid_pairs:
            Tn = p.get("T_ab_recomputed", p.get("T_ab_native", torch.eye(4)))
            if not torch.is_tensor(Tn):
                Tn = torch.tensor(Tn, dtype=torch.float32)
            Tn = Tn.float()

            do_perturb = bool(self.perturb_pair_transform and rng.random() < self.pair_perturb_prob and p.get("pair_valid", False))
            if do_perturb:
                Tin, _ = perturb_transform(Tn, max_angle_degrees=self.pair_max_rotation_degrees, max_translation=self.pair_max_translation)
                was = True
            else:
                Tin = Tn.clone()
                was = False

            Tres = compose_transform(Tn, inverse_transform(Tin))
            pair_T_native.append(Tn)
            pair_T_input.append(Tin)
            pair_T_target_residual.append(Tres)
            pair_was_perturbed.append(was)

        sample.update({
            "pair_features": pair_features,
            "pair_mask": pair_mask,
            "pair_T_input": torch.stack(pair_T_input, dim=0),
            "pair_T_native": torch.stack(pair_T_native, dim=0),
            "pair_T_target_residual": torch.stack(pair_T_target_residual, dim=0),
            "pair_ids": pair_ids,
            "pair_types": pair_types,
            "pair_valid": pair_valid,
            "pair_was_perturbed": torch.tensor(pair_was_perturbed, dtype=torch.bool),
        })
        return sample
