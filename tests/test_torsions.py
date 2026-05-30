import math

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from motordock.chem.torsions import find_rotatable_bonds
from motordock.diffusion.torsion import apply_torsion_updates, wrap_angle


def _coords_from_mol(mol):
    conf = mol.GetConformer()
    out = []
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        out.append([float(p.x), float(p.y), float(p.z)])
    return torch.tensor(out, dtype=torch.float32)


def test_butane_has_torsion_and_update_preserves_bond():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)

    bonds = find_rotatable_bonds(mol, mode="strict")
    assert len(bonds) >= 1

    rb = bonds[0]
    coords = _coords_from_mol(mol).unsqueeze(0)
    B, N = 1, coords.shape[1]

    mask = torch.tensor(rb.rotate_atom_mask, dtype=torch.bool).view(1, 1, N)
    bdict = {
        "atom_j": torch.tensor([[rb.atom_j]], dtype=torch.long),
        "atom_k": torch.tensor([[rb.atom_k]], dtype=torch.long),
    }
    delta = torch.tensor([[math.pi / 3]], dtype=torch.float32)  # +60 deg

    before = coords.clone()
    after = apply_torsion_updates(coords, bdict, mask, delta)

    assert not torch.allclose(before[:, mask[0, 0]], after[:, mask[0, 0]])

    jk_before = torch.linalg.norm(before[0, rb.atom_k] - before[0, rb.atom_j])
    jk_after = torch.linalg.norm(after[0, rb.atom_k] - after[0, rb.atom_j])
    assert torch.allclose(jk_before, jk_after, atol=1e-5, rtol=1e-5)


def test_benzene_has_no_strict_torsion():
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=0xBEEF)
    bonds = find_rotatable_bonds(mol, mode="strict")
    assert len(bonds) == 0


def test_wrap_angle_behavior():
    x = torch.tensor([math.pi + 0.1, -math.pi - 0.1], dtype=torch.float32)
    y = wrap_angle(x)
    assert torch.allclose(y[0], torch.tensor(-math.pi + 0.1), atol=1e-4)
    assert torch.allclose(y[1], torch.tensor(math.pi - 0.1), atol=1e-4)
