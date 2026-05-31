from __future__ import annotations

from pathlib import Path
import torch
import random
from rdkit import Chem, RDLogger, rdBase
RDLogger.DisableLog("rdApp.*")
rdBase.DisableLog("rdApp.*")

_HYB = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
_CHIRAL = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]


def _one_hot(value, vocab):
    return [1.0 if value == x else 0.0 for x in vocab]


def load_ligand_mol(ligand_file: str, sanitize: bool = False) -> Chem.Mol | None:
    p = Path(ligand_file)
    mol = None
    sdf = p.with_suffix(".sdf")
    if sdf.exists():
        sup = Chem.SDMolSupplier(str(sdf), sanitize=sanitize, removeHs=False)
        mol = sup[0] if len(sup) > 0 else None
    if mol is None:
        mol2 = p.with_suffix(".mol2")
        if mol2.exists():
            mol = Chem.MolFromMol2File(str(p), sanitize=sanitize, removeHs=False)
        if mol is None:
            mol = Chem.MolFromMol2File(str(p), sanitize=False, removeHs=False)
    return mol


def get_ligand_coordinates(mol: Chem.Mol) -> torch.Tensor:
    conf = mol.GetConformer()
    coords = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append([float(pos.x), float(pos.y), float(pos.z)])
    return torch.tensor(coords, dtype=torch.float32)


def featurize_ligand_mol(mol: Chem.Mol) -> torch.Tensor:
    feats = []
    for atom in mol.GetAtoms():
        f = [
            atom.GetAtomicNum() / 100.0,
            float(atom.GetDegree()),
            float(atom.GetFormalCharge()),
            float(atom.GetTotalValence()),
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
        ]
        f.extend(_one_hot(atom.GetHybridization(), _HYB))
        f.extend(_one_hot(atom.GetChiralTag(), _CHIRAL))
        f.append(float(atom.GetTotalNumHs(includeNeighbors=True)))
        f.append(atom.GetMass() * 0.01)
        feats.append(f)
    return torch.tensor(feats, dtype=torch.float32)

def random_truncate(mol, max_atoms):
    num_atoms = mol.GetNumAtoms()
    if num_atoms <= max_atoms:
        return mol
    else:
        max_start = num_atoms - max_atoms
        random_start = random.randint(0, max_start)
        keep_atoms = set(range(random_start, random_start + max_atoms))
        rw_mol = Chem.RWMol(mol)
        for idx in range(num_atoms - 1, -1, -1):
            if idx not in keep_atoms:
                rw_mol.RemoveAtom(idx)
        return rw_mol.GetMol()