from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import Dataset

from .ligand_featurizer import load_ligand_mol, get_ligand_coordinates, featurize_ligand_mol
from .residue_featurizer import featurize_protein_sequence
from .pose_noise import randomize_ligand_pose
from motordock.chem.torsions import (
    find_rotatable_bonds,
    assert_atom_count_matches_coords,
    torsion_angle_from_coords,
)


@dataclass
class SampleError(Exception):
    pdb_id: str
    reason: str

    def __str__(self):
        return f"[{self.pdb_id}] {self.reason}"


class PDBBindBaselineDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        output_dir: str,
        split: str | None = None,
        max_examples: int | None = None,
        require_pocket: bool = True,
        max_ligand_atoms: int = 128,
        max_protein_residues: int = 1022,
        sanitize_ligand: bool = True,
        randomize_pose: bool = True,
        max_translation: float = 10.0,
        max_rotation_degrees: float = 180.0,
        seed: int = 0,
        prevalidate: bool = True,
        debug_raise: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.require_pocket = require_pocket
        self.max_ligand_atoms = max_ligand_atoms
        self.max_protein_residues = max_protein_residues
        self.sanitize_ligand = sanitize_ligand
        self.randomize_pose = randomize_pose
        self.max_translation = max_translation
        self.max_rotation_degrees = max_rotation_degrees
        self.seed = seed

        df = pd.read_csv(csv_path)
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split]
        if max_examples is not None:
            df = df.head(max_examples)
        self.df = df.reset_index(drop=True)

        self.valid_indices = list(range(len(self.df)))
        if prevalidate:
            ok = []
            for i in range(len(self.df)):
                try:
                    self._load_raw(i, validate_only=True)
                    ok.append(i)
                except Exception as e:
                    if debug_raise:
                        raise
            self.valid_indices = ok

    def __len__(self):
        return len(self.valid_indices)

    def _artifact(self, sub: str, pdb_id: str, suffix: str):
        return self.output_dir / sub / f"{pdb_id}_{suffix}.pt"

    def _load_raw(self, idx: int, validate_only: bool = False):
        row = self.df.iloc[idx]
        pdb_id = str(row["Assembly_ID"])

        emb = torch.load(self._artifact("embeddings", pdb_id, "prot_emb"), map_location="cpu")
        pocket_mask = torch.load(self._artifact("pocket_masks", pdb_id, "pocket_mask"), map_location="cpu")
        domain_mask = torch.load(self._artifact("domain_masks", pdb_id, "domain_mask"), map_location="cpu")
        pocket_info = torch.load(self._artifact("pocket_info", pdb_id, "pocket_info"), map_location="cpu")

        protein_ca = emb["ca_coord"].float()
        protein_seq = emb["sequence"]
        if protein_ca.shape[0] > self.max_protein_residues:
            protein_ca = protein_ca[: self.max_protein_residues]
            protein_seq = protein_seq[: self.max_protein_residues]
            pocket_mask = pocket_mask[: self.max_protein_residues]
            domain_mask = domain_mask[: self.max_protein_residues]

        pocket_center = pocket_info.get("pocket_center_ca", None)
        if pocket_center is None:
            pocket_center = pocket_info.get("pocket_center_atom", None)
        if pocket_center is None:
            if self.require_pocket:
                raise SampleError(pdb_id, "missing pocket center")
            pocket_center = protein_ca.mean(dim=0)
        else:
            pocket_center = pocket_center.float()

        lig_file = str(row["ligand_file"])
        mol = load_ligand_mol(lig_file, sanitize=self.sanitize_ligand)
        if mol is None:
            raise SampleError(pdb_id, f"failed ligand parsing: {lig_file}")

        lig_coords_true = get_ligand_coordinates(mol).float()
        assert_atom_count_matches_coords(mol, lig_coords_true)
        if lig_coords_true.shape[0] > self.max_ligand_atoms:
            raise SampleError(pdb_id, f"too many ligand atoms: {lig_coords_true.shape[0]}")
        lig_feat = featurize_ligand_mol(mol).float()

        rot_bonds = find_rotatable_bonds(mol, mode="strict")
        M = len(rot_bonds)
        torsion_bond_atom_j = torch.zeros((M,), dtype=torch.long)
        torsion_bond_atom_k = torch.zeros((M,), dtype=torch.long)
        torsion_atom_mask = torch.zeros((M, lig_coords_true.shape[0]), dtype=torch.bool)
        torsion_valid_mask = torch.ones((M,), dtype=torch.bool)
        torsion_angles_0 = torch.zeros((M,), dtype=torch.float32)
        for m, rb in enumerate(rot_bonds):
            torsion_bond_atom_j[m] = int(rb.atom_j)
            torsion_bond_atom_k[m] = int(rb.atom_k)
            torsion_atom_mask[m] = torch.tensor(rb.rotate_atom_mask, dtype=torch.bool)
            torsion_angles_0[m] = torsion_angle_from_coords(
                lig_coords_true,
                rb.atom_i,
                rb.atom_j,
                rb.atom_k,
                rb.atom_l,
            )

        if validate_only:
            return None

        g = torch.Generator()
        g.manual_seed(self.seed + idx)
        _ = g

        if self.randomize_pose:
            lig_coords_start, T_noise, T_target = randomize_ligand_pose(
                lig_coords_true,
                pocket_center,
                max_translation=self.max_translation,
                max_rotation_degrees=self.max_rotation_degrees,
            )
        else:
            lig_coords_start = lig_coords_true.clone()
            T_noise = torch.eye(4, dtype=torch.float32)
            T_target = torch.eye(4, dtype=torch.float32)

        protein_feat = featurize_protein_sequence(protein_seq, pocket_mask, domain_mask)

        return {
            "pdb_id": pdb_id,
            "split": str(row.get("split", "unknown")),
            "motordock_type": str(row.get("motordock_type", "Unknown")),
            "protein_file": str(row.get("protein_file", "")),
            "ligand_file": lig_file,
            "pocket_file": str(row.get("pocket_file", "")),
            "protein_seq": protein_seq,
            "protein_ca": protein_ca,
            "protein_feat": protein_feat,
            "protein_mask": torch.ones(protein_ca.shape[0], dtype=torch.bool),
            "pocket_mask": pocket_mask.long(),
            "domain_mask": domain_mask.long(),
            "pocket_center": pocket_center,
            "ligand_atom_feat": lig_feat,
            "ligand_coords_true": lig_coords_true,
            "ligand_coords_start": lig_coords_start.float(),
            "ligand_mask": torch.ones(lig_coords_true.shape[0], dtype=torch.bool),
            "T_noise": T_noise.float(),
            "T_target": T_target.float(),
            "torsion_bond_atom_j": torsion_bond_atom_j,
            "torsion_bond_atom_k": torsion_bond_atom_k,
            "torsion_atom_mask": torsion_atom_mask,
            "torsion_valid_mask": torsion_valid_mask,
            "torsion_angles_0": torsion_angles_0,
            "affinity": float(row.get("affinity", 0.0)),
        }

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        return self._load_raw(real_idx, validate_only=False)
