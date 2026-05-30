from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader

from motordock.data.pdbbind_baseline_dataset import PDBBindBaselineDataset
from motordock.data.collate import baseline_collate_fn
from motordock.models.diffusion_docking_model import DiffusionDockingModel
from motordock.train.validate_diffusion_baseline import (
    validate_diffusion_loss,
    validate_diffusion_sampling,
)


def _tiny_dataset(tmp_path: Path):
    out = tmp_path / "data"
    for d in ["embeddings", "domain_masks", "pocket_masks", "pocket_info"]:
        (out / d).mkdir(parents=True)
    pid = "xxxx"
    torch.save({"ca_coord": torch.randn(8, 3), "sequence": "ACDEFGHI"}, out / "embeddings" / f"{pid}_prot_emb.pt")
    torch.save(torch.ones(8, dtype=torch.long), out / "domain_masks" / f"{pid}_domain_mask.pt")
    torch.save(torch.ones(8, dtype=torch.long), out / "pocket_masks" / f"{pid}_pocket_mask.pt")
    torch.save({"pocket_center_ca": torch.zeros(3), "pocket_center_atom": None}, out / "pocket_info" / f"{pid}_pocket_info.pt")

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    sdf = tmp_path / "lig.sdf"
    w = Chem.SDWriter(str(sdf))
    w.write(mol)
    w.close()

    df = pd.DataFrame([
        {
            "Assembly_ID": pid,
            "split": "val",
            "motordock_type": "Single-Domain",
            "protein_file": "p.pdb",
            "ligand_file": str(sdf),
            "pocket_file": "pk.pdb",
            "affinity": 0.0,
        }
    ])
    csv = tmp_path / "pdbbind_ft.csv"
    df.to_csv(csv, index=False)

    ds = PDBBindBaselineDataset(str(csv), str(out), split="val", randomize_pose=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=baseline_collate_fn)
    return ds, dl


def _cfg():
    return {
        "diffusion": {
            "num_steps": 3,
            "sigma_tr_min": 0.1,
            "sigma_tr_max": 1.0,
            "sigma_rot_min": 0.05,
            "sigma_rot_max": 0.5,
            "schedule_type": "log_linear",
            "deterministic_eval": True,
            "init_translation_sigma": 1.0,
        },
        "train_diffusion": {
            "lambda_tr": 1.0,
            "lambda_rot": 1.0,
            "sampling_val_num_samples": 2,
        },
    }


def test_validate_diffusion_loss_returns_finite(tmp_path: Path):
    ds, dl = _tiny_dataset(tmp_path)
    s = ds[0]
    m = DiffusionDockingModel(s["protein_feat"].shape[-1], s["ligand_atom_feat"].shape[-1], hidden_dim=32, num_layers=2, dropout=0.0)
    out = validate_diffusion_loss(m, dl, torch.device("cpu"), _cfg())
    assert "val_diffusion_loss" in out
    assert torch.isfinite(torch.tensor(out["val_diffusion_loss"]))


def test_validate_diffusion_sampling_keys_and_shapes(tmp_path: Path):
    ds, dl = _tiny_dataset(tmp_path)
    s = ds[0]
    m = DiffusionDockingModel(s["protein_feat"].shape[-1], s["ligand_atom_feat"].shape[-1], hidden_dim=32, num_layers=2, dropout=0.0)
    out = validate_diffusion_sampling(m, dl, torch.device("cpu"), _cfg(), num_samples=2)
    for k in [
        "val_top1_by_confidence_rmsd",
        "val_oracle_topk_rmsd",
        "val_top1_success_2A",
        "val_oracle_topk_success_2A",
    ]:
        assert k in out
        assert torch.isfinite(torch.tensor(out[k]))
