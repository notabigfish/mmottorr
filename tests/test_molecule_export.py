from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from motordock.eval.molecule_export import export_predicted_ligand_sdf, prepare_posebusters_table


def _make_ethanol_sdf(path: Path):
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    w = Chem.SDWriter(str(path))
    w.write(mol)
    w.close()
    return mol


def _coords_from_mol(mol):
    conf = mol.GetConformer()
    out = []
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        out.append([float(p.x), float(p.y), float(p.z)])
    return torch.tensor(out, dtype=torch.float32)


def test_export_predicted_sdf(tmp_path: Path):
    template = tmp_path / "template.sdf"
    mol = _make_ethanol_sdf(template)
    coords = _coords_from_mol(mol)
    shifted = coords + torch.tensor([1.0, -2.0, 0.5])

    out_sdf = tmp_path / "pred.sdf"
    export_predicted_ligand_sdf(str(template), shifted, str(out_sdf), sanitize=True)

    sup = Chem.SDMolSupplier(str(out_sdf), sanitize=True, removeHs=False)
    out_mol = sup[0]
    assert out_mol is not None
    assert out_mol.GetNumAtoms() == mol.GetNumAtoms()
    assert out_mol.GetNumBonds() == mol.GetNumBonds()

    out_coords = _coords_from_mol(out_mol)
    assert torch.allclose(out_coords, shifted, atol=1e-4, rtol=1e-4)


def test_prepare_posebusters_table(tmp_path: Path):
    template = tmp_path / "lig_true.sdf"
    mol = _make_ethanol_sdf(template)
    coords = _coords_from_mol(mol)

    protein = tmp_path / "prot.pdb"
    protein.write_text("HEADER\nEND\n", encoding="utf-8")

    pred1 = tmp_path / "pred1.pt"
    pred2 = tmp_path / "pred2.pt"
    torch.save(coords + 0.1, pred1)
    torch.save({"ligand_coords_pred": coords + 0.2}, pred2)

    df = pd.DataFrame(
        [
            {
                "complex_id": "c1",
                "rank": 0,
                "protein_path": str(protein),
                "ligand_true_path": str(template),
                "ligand_template_path": str(template),
                "pred_coords_path": str(pred1),
            },
            {
                "complex_id": "c1",
                "rank": 1,
                "protein_path": str(protein),
                "ligand_true_path": str(template),
                "ligand_template_path": str(template),
                "pred_coords_path": str(pred2),
            },
        ]
    )
    csv = tmp_path / "pred.csv"
    df.to_csv(csv, index=False)

    out = prepare_posebusters_table(str(csv), str(tmp_path / "exports"))
    assert list(out.columns) == ["mol_pred", "mol_true", "mol_cond", "complex_id", "rank"]
    assert len(out) == 2
    for p in out["mol_pred"].tolist():
        assert Path(p).exists()
