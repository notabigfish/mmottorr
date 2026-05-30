from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from motordock.train.train_diffusion_baseline import train_diffusion_baseline
from motordock.train.checkpointing import load_checkpoint


def _make_tiny_artifacts(tmp_path: Path):
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
            "split": "train",
            "motordock_type": "Single-Domain",
            "protein_file": "p.pdb",
            "ligand_file": str(sdf),
            "pocket_file": "pk.pdb",
            "affinity": 0.0,
        },
        {
            "Assembly_ID": pid,
            "split": "val",
            "motordock_type": "Single-Domain",
            "protein_file": "p.pdb",
            "ligand_file": str(sdf),
            "pocket_file": "pk.pdb",
            "affinity": 0.0,
        },
    ])
    csv = tmp_path / "pdbbind_ft.csv"
    df.to_csv(csv, index=False)
    return out, csv


def test_train_diffusion_smoke(tmp_path: Path):
    out_dir, csv_path = _make_tiny_artifacts(tmp_path)
    run_dir = tmp_path / "runs"
    cfg = {
        "seed": 42,
        "data": {
            "csv_path": str(csv_path),
            "output_dir": str(out_dir),
            "split_train": "train",
            "split_val": "val",
            "max_train_examples": 1,
            "max_val_examples": 1,
            "require_pocket": True,
            "max_ligand_atoms": 128,
            "max_protein_residues": 1022,
        },
        "model": {"hidden_dim": 32, "num_layers": 2, "dropout": 0.0},
        "train": {
            "batch_size": 1,
            "num_workers": 0,
            "epochs": 1,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
            "use_amp": False,
            "log_interval": 1,
            "val_interval": 1,
        },
        "pose_noise": {"max_translation": 1.0, "max_rotation_degrees": 30.0},
        "diffusion": {
            "num_steps": 3,
            "sigma_tr_min": 0.1,
            "sigma_tr_max": 1.0,
            "sigma_rot_min": 0.05,
            "sigma_rot_max": 0.5,
            "schedule_type": "log_linear",
            "deterministic_eval": True,
            "init_translation_sigma": 1.0,
            "center_init": "pocket",
        },
        "train_diffusion": {
            "lambda_tr": 1.0,
            "lambda_rot": 1.0,
            "val_mode": "dual",
            "sampling_val_num_samples": 2,
        },
        "output": {"run_dir": str(run_dir), "save_every": 1},
    }

    res = train_diffusion_baseline(cfg)
    assert "best_val_metric" in res

    latest = run_dir / "latest.pt"
    best = run_dir / "best.pt"
    log_csv = run_dir / "train_log.csv"

    assert latest.exists()
    assert best.exists()
    assert log_csv.exists()

    ck = load_checkpoint(str(best), map_location="cpu")
    assert ck["model_type"] == "diffusion_baseline"
