from pathlib import Path
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from motordock.data.pdbbind_baseline_dataset import PDBBindBaselineDataset


def test_dataset_loads_one_sample(tmp_path: Path):
    out = tmp_path / "data"
    for d in ["embeddings", "domain_masks", "pocket_masks", "pocket_info"]:
        (out / d).mkdir(parents=True)
    pid = "xxxx"
    torch.save({"ca_coord": torch.randn(8, 3), "sequence": "ACDEFGHI"}, out / "embeddings" / f"{pid}_prot_emb.pt")
    torch.save(torch.ones(8, dtype=torch.long), out / "domain_masks" / f"{pid}_domain_mask.pt")
    torch.save(torch.ones(8, dtype=torch.long), out / "pocket_masks" / f"{pid}_pocket_mask.pt")
    torch.save({"pocket_center_ca": torch.zeros(3), "pocket_center_atom": None}, out / "pocket_info" / f"{pid}_pocket_info.pt")

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
    sdf = tmp_path / "lig.sdf"
    w = Chem.SDWriter(str(sdf)); w.write(mol); w.close()

    df = pd.DataFrame([{
        "Assembly_ID": pid,
        "split": "train",
        "motordock_type": "Single-Domain",
        "protein_file": "p.pdb",
        "ligand_file": str(sdf),
        "pocket_file": "pk.pdb",
        "affinity": 0.0,
    }])
    csv = tmp_path / "pdbbind_ft.csv"
    df.to_csv(csv, index=False)

    ds = PDBBindBaselineDataset(str(csv), str(out), split="train", randomize_pose=False)
    s = ds[0]
    assert s["protein_ca"].shape[0] == 8
    assert s["ligand_coords_true"].ndim == 2
