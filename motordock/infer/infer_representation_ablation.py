from __future__ import annotations

from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from motordock.train.checkpointing import load_checkpoint
from motordock.data.ablation_dataset import RepresentationAblationDataset
from motordock.data.motordock_collate import motordock_se3_collate_fn
from motordock.data.representation_pair_featurizer import representation_pair_feature_dim
from motordock.models.motordock_ablation_model import MotorDockAblationModel
from motordock.eval.metrics_pose import ligand_rmsd, centroid_distance
from motordock.eval.metrics_pair import attention_entropy


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}


def run_inference(checkpoint: str, config: dict, split: str, num_samples: int, out_csv: str):
    ck = load_checkpoint(checkpoint, map_location="cpu")
    d,m,r,pn = config["data"], config["model"], config["representation"], config["pose_noise"]
    ablation_mode = r["name"] if r["name"] in {"random_motor","shuffled_pairs","no_pair_context"} else "normal"

    ds = RepresentationAblationDataset(d["csv_path"], d["output_dir"], split=split, max_examples=d.get("max_val_examples"), require_pocket=d.get("require_pocket",True), max_ligand_atoms=d.get("max_ligand_atoms",128), max_protein_residues=d.get("max_protein_residues",1022), max_candidate_pairs=d.get("max_candidate_pairs",16), randomize_pose=True, max_translation=pn["max_translation"], max_rotation_degrees=pn["max_rotation_degrees"], perturb_pair_transform=False, pair_perturb_prob=0.0, representation=r["name"], matrix_mode=r.get("matrix_mode","3x4"), ablation_mode=ablation_mode, shuffle_pair_features=r.get("shuffle_pair_features",False), random_motor_seed=r.get("random_motor_seed",0))
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=motordock_se3_collate_fn)
    s = ds[0]
    model = MotorDockAblationModel(s["protein_feat"].shape[-1], s["ligand_atom_feat"].shape[-1], representation_pair_feature_dim(r["name"], r.get("matrix_mode","3x4")), representation=r["name"], matrix_mode=r.get("matrix_mode","3x4"), hidden_dim=m["hidden_dim"], num_layers=m["num_layers"], dropout=m["dropout"], use_pair_attention=m.get("use_pair_attention",True), disable_pair_context=m.get("disable_pair_context",False), parameter_budget_mode=r.get("parameter_budget_mode","matched"), max_rotation_scale=r.get("max_rotation_scale",0.5), max_translation_scale=r.get("max_translation_scale",5.0))
    model.load_state_dict(ck["model_state_dict"])

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev); model.eval()
    pred_dir = Path(out_csv).parent / "predictions"; pred_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.no_grad():
        for batch in dl:
            for i in range(num_samples):
                b = _to_device(batch, dev)
                o = model(b)
                rmsd = ligand_rmsd(o["ligand_coords_pred"], b["ligand_coords_true"], b["ligand_mask"])[0].item()
                cent = centroid_distance(o["ligand_coords_pred"], b["ligand_coords_true"], b["ligand_mask"])[0].item()
                conf = o["confidence_logit"][0].item()
                attn = o["pair_attention"][0].detach().cpu()
                rank = int(torch.argmax(attn).item())
                pid = batch["pdb_id"][0]
                ppath = pred_dir / f"{pid}_sample{i}.pt"
                torch.save(
                    {
                        "pdb_id": pid, 
                        "representation": r["name"], 
                        "ligand_coords_pred": o["ligand_coords_pred"][0].detach().cpu(), 
                        "ligand_coords_true": b["ligand_coords_true"][0].detach().cpu(), 
                        "ligand_coords_start": b["ligand_coords_start"][0].detach().cpu(), 
                        "confidence": float(conf), 
                        "rmsd": float(rmsd), 
                        "pair_attention": attn, 
                        "pair_ids": batch["pair_ids"][0], 
                        "pair_types": batch["pair_types"][0], 
                        "pair_T_input": b["pair_T_input"][0].detach().cpu(), 
                        "pair_T_corrected": o["pair_T_corrected"][0].detach().cpu()
                    }, ppath)
                rows.append(
                    {
                        "pdb_id": pid, 
                        "split": batch["split"][0], 
                        "motordock_type": batch["motordock_type"][0], 
                        "representation": r["name"], 
                        "sample_idx": i, 
                        "confidence": conf, 
                        "rmsd": rmsd, 
                        "centroid_distance": cent, 
                        "success_2A": int(rmsd < 2.0), 
                        "selected_pair_rank": rank, 
                        "selected_pair_id": batch["pair_ids"][0][rank] if rank < len(batch["pair_ids"][0]) else "identity_fallback", 
                        "selected_pair_type": batch["pair_types"][0][rank] if rank < len(batch["pair_types"][0]) else "unknown", 
                        "selected_pair_attention": float(attn[rank].item()), 
                        "mean_pair_attention_entropy": float(attention_entropy(attn.unsqueeze(0), b["pair_mask"][0].detach().cpu().unsqueeze(0)).item()), 
                        "protein_file": batch["protein_file"][0], 
                        "ligand_file": batch["ligand_file"][0], 
                        "pocket_file": batch["pocket_file"][0], 
                        "prediction_path": str(ppath)
                    })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv
