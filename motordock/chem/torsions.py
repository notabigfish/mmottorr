from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from rdkit import Chem


@dataclass
class RotatableBond:
    bond_index: int
    atom_j: int
    atom_k: int
    atom_i: int
    atom_l: int
    rotate_atom_mask: list[bool]


def load_ligand_mol(path: str, sanitize: bool = True) -> Chem.Mol:
    p = Path(path)
    mol = None

    if p.suffix.lower() == ".sdf" and p.exists():
        sup = Chem.SDMolSupplier(str(p), sanitize=sanitize, removeHs=False)
        mol = sup[0] if len(sup) > 0 else None
    elif p.suffix.lower() == ".mol2" and p.exists():
        mol = Chem.MolFromMol2File(str(p), sanitize=sanitize, removeHs=False)

    if mol is None:
        sdf = p.with_suffix(".sdf")
        if sdf.exists():
            sup = Chem.SDMolSupplier(str(sdf), sanitize=sanitize, removeHs=False)
            mol = sup[0] if len(sup) > 0 else None
    if mol is None:
        mol2 = p.with_suffix(".mol2")
        if mol2.exists():
            mol = Chem.MolFromMol2File(str(mol2), sanitize=sanitize, removeHs=False)

    if mol is None:
        raise ValueError(f"failed to load ligand molecule from: {path}")
    return mol


def get_heavy_atom_order(mol) -> list[int]:
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]


def assert_atom_count_matches_coords(mol, coords):
    n_atoms = mol.GetNumAtoms()
    n_coords = int(coords.shape[0])
    if n_atoms != n_coords:
        raise ValueError(f"atom count mismatch: mol={n_atoms} coords={n_coords}")


def _is_amide_cn(bond: Chem.Bond) -> bool:
    a = bond.GetBeginAtom()
    b = bond.GetEndAtom()
    if {a.GetAtomicNum(), b.GetAtomicNum()} != {6, 7}:
        return False
    c_atom = a if a.GetAtomicNum() == 6 else b
    for nb in c_atom.GetBonds():
        if nb.GetIdx() == bond.GetIdx():
            continue
        if nb.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            other = nb.GetOtherAtom(c_atom)
            if other.GetAtomicNum() == 8:
                return True
    return False


def _adj_list_without_bond(mol: Chem.Mol, drop_bond_idx: int):
    n = mol.GetNumAtoms()
    adj = [[] for _ in range(n)]
    for b in mol.GetBonds():
        if b.GetIdx() == drop_bond_idx:
            continue
        u = b.GetBeginAtomIdx()
        v = b.GetEndAtomIdx()
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _component_from(start: int, adj: list[list[int]]):
    seen = set([start])
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def _heavy_count(mol: Chem.Mol, atom_ids: set[int]) -> int:
    c = 0
    for i in atom_ids:
        if mol.GetAtomWithIdx(i).GetAtomicNum() > 1:
            c += 1
    return c


def _pick_neighbor(atom: Chem.Atom, avoid_idx: int) -> int | None:
    for nb in atom.GetNeighbors():
        if nb.GetIdx() != avoid_idx:
            return nb.GetIdx()
    return None


def find_rotatable_bonds(mol: Chem.Mol, mode: str = "strict", allow_terminal: bool = False) -> list[RotatableBond]:
    if mode != "strict":
        raise ValueError(f"unsupported mode: {mode}")

    n_atoms = mol.GetNumAtoms()
    out: list[RotatableBond] = []

    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
            continue
        if bond.IsInRing():
            continue

        j = bond.GetBeginAtomIdx()
        k = bond.GetEndAtomIdx()
        aj = mol.GetAtomWithIdx(j)
        ak = mol.GetAtomWithIdx(k)

        if aj.GetAtomicNum() <= 1 or ak.GetAtomicNum() <= 1:
            continue

        if _is_amide_cn(bond):
            continue

        if not allow_terminal:
            if aj.GetDegree() <= 1 or ak.GetDegree() <= 1:
                continue

        adj = _adj_list_without_bond(mol, bond.GetIdx())
        comp_j = _component_from(j, adj)
        comp_k = _component_from(k, adj)

        hj = _heavy_count(mol, comp_j)
        hk = _heavy_count(mol, comp_k)
        rotate_side = comp_j if hj <= hk else comp_k

        if not allow_terminal and _heavy_count(mol, rotate_side) < 2:
            continue

        i = _pick_neighbor(aj, k)
        l = _pick_neighbor(ak, j)
        if i is None or l is None:
            continue

        rotate_atom_mask = [False] * n_atoms
        for aidx in rotate_side:
            rotate_atom_mask[aidx] = True

        out.append(
            RotatableBond(
                bond_index=bond.GetIdx(),
                atom_j=j,
                atom_k=k,
                atom_i=i,
                atom_l=l,
                rotate_atom_mask=rotate_atom_mask,
            )
        )

    return out


def torsion_angle_from_coords(coords: torch.Tensor, i: int, j: int, k: int, l: int) -> torch.Tensor:
    b1 = coords[j] - coords[i]
    b2 = coords[k] - coords[j]
    b3 = coords[l] - coords[k]

    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)

    n1 = n1 / n1.norm().clamp_min(1e-12)
    n2 = n2 / n2.norm().clamp_min(1e-12)
    b2u = b2 / b2.norm().clamp_min(1e-12)

    x = (n1 * n2).sum()
    y = (torch.cross(n1, n2, dim=-1) * b2u).sum()
    return torch.atan2(y, x)
