from pathlib import Path
import builtins

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from motordock.eval.posebusters_runner import run_posebusters_if_available


def _make_inputs(tmp_path: Path):
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    lig = tmp_path / "lig.sdf"
    w = Chem.SDWriter(str(lig)); w.write(mol); w.close()

    conf = mol.GetConformer()
    coords = []
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        coords.append([float(p.x), float(p.y), float(p.z)])
    coords = torch.tensor(coords)

    pred = tmp_path / "pred.pt"
    torch.save(coords + 0.3, pred)

    prot = tmp_path / "prot.pdb"
    prot.write_text("HEADER\nEND\n", encoding="utf-8")

    csv = tmp_path / "pred.csv"
    pd.DataFrame([
        {
            "complex_id": "cx",
            "rank": 0,
            "protein_path": str(prot),
            "ligand_true_path": str(lig),
            "ligand_template_path": str(lig),
            "pred_coords_path": str(pred),
        }
    ]).to_csv(csv, index=False)
    return csv


def test_posebusters_unavailable(monkeypatch, tmp_path: Path):
    csv = _make_inputs(tmp_path)
    out = tmp_path / "pb.csv"

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "posebusters":
            raise ImportError("no posebusters")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    res = run_posebusters_if_available(str(csv), str(out))
    assert res["available"] is False
    assert "not installed" in res["reason"]


def test_posebusters_mocked(monkeypatch, tmp_path: Path):
    csv = _make_inputs(tmp_path)
    out_csv = tmp_path / "pb_out.csv"

    class FakePB:
        def __init__(self, config="redock", top_n=None, max_workers=0):
            self.config = config

        def bust_table(self, table, full_report=True):
            return pd.DataFrame(
                {
                    "sanitization": [True] * len(table),
                    "bond_lengths": [True] * len(table),
                    "volume_overlap_with_protein": [False] * len(table),
                    "rmsd_≤_2å": [True] * len(table),
                }
            )

    class FakeModule:
        PoseBusters = FakePB

    import sys
    monkeypatch.setitem(sys.modules, "posebusters", FakeModule())

    res = run_posebusters_if_available(
        str(csv),
        str(out_csv),
        export_dir=str(tmp_path / "exported"),
        config="redock",
        full_report=True,
    )

    assert res["available"] is True
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert "pb_valid" in df.columns
    assert "pb_valid_no_rmsd" in df.columns
    assert bool(df.loc[0, "pb_valid"]) is False
    assert bool(df.loc[0, "pb_valid_no_rmsd"]) is False
