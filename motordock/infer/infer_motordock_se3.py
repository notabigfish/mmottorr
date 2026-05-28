from __future__ import annotations

from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from motordock.data.motordock_dataset import MotorDockSE3Dataset
from motordock.data.motordock_collate import motordock_se3_collate_fn
from motordock.models.motordock_se3_model import MotorDockSE3Model
from motordock.data.pair_featurizer import pair_feature_dim
from motordock.eval.metrics_pose import ligand_rmsd, centroid_distance
from motordock.eval.metrics_pair import attention_entropy
from motordock.train.checkpointing import load_checkpoint


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def run_inference(checkpoint: str, csv_path: str, output_dir: str, split: str, num_samples: int, out_csv: str):
    ck = load_checkpoint(checkpoint, map_location="cpu")
    cfg = ck["config"]
    d = cfg["data"]
    pn = cfg["pose_noise"]

    ds = MotorDockSE3Dataset(
        csv_path, output_dir, split=split, max_examples=d.get("max_val_examples"), require_pocket=d.get("require_pocket", True),
        max_ligand_atoms=d.get("max_ligand_atoms", 128), max_protein_residues=d.get("max_protein_residues", 1022),
        max_candidate_pairs=d.get("max_candidate_pairs", 16), randomize_pose=True,
        max_translation=pn["max_translation"], max_rotation_degrees=pn["max_rotation_degrees"], perturb_pair_transform=False,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=motordock_se3_collate_fn)

    s = ds[0]
    mcfg = cfg["model"]
    model = MotorDockSE3Model(
        s["protein_feat"].shape[-1], s["ligand_atom_feat"].shape[-1], pair_feature_dim(),
        hidden_dim=mcfg["hidden_dim"], num_layers=mcfg["num_layers"], dropout=mcfg["dropout"],
        use_pair_attention=mcfg.get("use_pair_attention", True), use_motor_auxiliary=mcfg.get("use_motor_auxiliary", True),
        disable_pair_context=mcfg.get("disable_pair_context", False), freeze_baseline_encoder=mcfg.get("freeze_baseline_encoder", False),
        max_pair_rotation_scale=mcfg.get("max_pair_rotation_scale", 0.5), max_pair_translation_scale=mcfg.get("max_pair_translation_scale", 5.0),
    )
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
                attn = out["pair_attention"][0].detach().cpu()
                best = int(torch.argmax(attn).item())
                pdb_id = batch["pdb_id"][0]
                ppath = pred_dir / f"{pdb_id}_sample{sidx}.pt"
                torch.save({
                    "pdb_id": pdb_id,
                    "ligand_coords_pred": out["ligand_coords_pred"][0].detach().cpu(),
                    "ligand_coords_true": b["ligand_coords_true"][0].detach().cpu(),
                    "ligand_coords_start": b["ligand_coords_start"][0].detach().cpu(),
                    "confidence": float(conf),
                    "rmsd": float(r),
                    "pair_attention": attn,
                    "pair_ids": batch["pair_ids"][0],
                    "pair_types": batch["pair_types"][0],
                    "pair_T_input": b["pair_T_input"][0].detach().cpu(),
                    "pair_T_corrected": out["pair_T_corrected"][0].detach().cpu(),
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
                    "selected_pair_rank": best,
                    "selected_pair_id": batch["pair_ids"][0][best] if best < len(batch["pair_ids"][0]) else "identity_fallback",
                    "selected_pair_type": batch["pair_types"][0][best] if best < len(batch["pair_types"][0]) else "unknown",
                    "selected_pair_attention": float(attn[best].item()),
                    "mean_pair_attention_entropy": float(attention_entropy(attn.unsqueeze(0), b["pair_mask"][0].detach().cpu().unsqueeze(0)).item()),
                    "protein_file": batch["protein_file"][0],
                    "ligand_file": batch["ligand_file"][0],
                    "pocket_file": batch["pocket_file"][0],
                    "prediction_path": str(ppath),
                })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv
