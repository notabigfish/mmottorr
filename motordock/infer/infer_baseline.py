from __future__ import annotations

from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from motordock.data import PDBBindBaselineDataset, baseline_collate_fn
from motordock.models import BaselineDockingModel
from motordock.eval.metrics_pose import ligand_rmsd, centroid_distance
from motordock.train.checkpointing import load_checkpoint


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def run_inference(checkpoint: str, csv_path: str, output_dir: str, split: str, num_samples: int, out_csv: str):
    ck = load_checkpoint(checkpoint, map_location="cpu")
    cfg = ck["config"]
    dcfg = cfg["data"]
    ncfg = cfg["pose_noise"]

    ds = PDBBindBaselineDataset(
        csv_path=csv_path, output_dir=output_dir, split=split, max_examples=dcfg.get("max_val_examples"),
        require_pocket=dcfg.get("require_pocket", True), max_ligand_atoms=dcfg.get("max_ligand_atoms", 128),
        max_protein_residues=dcfg.get("max_protein_residues", 1022), randomize_pose=True,
        max_translation=ncfg["max_translation"], max_rotation_degrees=ncfg["max_rotation_degrees"],
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=baseline_collate_fn)

    sample = ds[0]
    model = BaselineDockingModel(sample["protein_feat"].shape[-1], sample["ligand_atom_feat"].shape[-1], cfg["model"]["hidden_dim"], cfg["model"]["num_layers"], cfg["model"]["dropout"])
    model.load_state_dict(ck["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    pred_dir = Path(out_csv).parent / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.no_grad():
        for batch in dl:
            for sidx in range(num_samples):
                b = _to_device(batch, device)
                out = model(b)
                r = ligand_rmsd(out["ligand_coords_pred"], b["ligand_coords_true"], b["ligand_mask"])[0].item()
                c = centroid_distance(out["ligand_coords_pred"], b["ligand_coords_true"], b["ligand_mask"])[0].item()
                conf = out["confidence_logit"][0].item()
                pdb_id = batch["pdb_id"][0]
                ppath = pred_dir / f"{pdb_id}_sample{sidx}.pt"
                torch.save({
                    "pdb_id": pdb_id,
                    "ligand_coords_pred": out["ligand_coords_pred"][0].detach().cpu(),
                    "ligand_coords_true": b["ligand_coords_true"][0].detach().cpu(),
                    "ligand_coords_start": b["ligand_coords_start"][0].detach().cpu(),
                    "confidence": float(conf),
                    "rmsd": float(r),
                }, ppath)
                rows.append({
                    "pdb_id": pdb_id,
                    "split": batch["split"][0],
                    "motordock_type": batch["motordock_type"][0],
                    "sample_idx": sidx,
                    "confidence": conf,
                    "rmsd": r,
                    "centroid_distance": c,
                    "success_2A": int(r < 2.0),
                    "protein_file": batch["protein_file"][0],
                    "ligand_file": batch["ligand_file"][0],
                    "pocket_file": batch["pocket_file"][0],
                    "prediction_path": str(ppath),
                })

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv
