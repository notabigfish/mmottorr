from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader

from motordock.data.motordock_dataset import MotorDockSE3Dataset
from motordock.data.motordock_collate import motordock_se3_collate_fn
from motordock.models import MotorDockDiffusionModel
from motordock.train.checkpointing import load_checkpoint
from motordock.diffusion.motordock_diffusion_sampler import sample_motordock_diffusion


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    ck = load_checkpoint(args.checkpoint, map_location="cpu")
    d = cfg["data"]; pn = cfg["pose_noise"]
    ds = MotorDockSE3Dataset(d["csv_path"], d["output_dir"], split=args.split, max_examples=d.get("max_val_examples"),
                             require_pocket=d.get("require_pocket", True), max_ligand_atoms=d.get("max_ligand_atoms", 128),
                             max_protein_residues=d.get("max_protein_residues", 1022), max_candidate_pairs=d.get("max_candidate_pairs", 16),
                             randomize_pose=True, max_translation=pn["max_translation"], max_rotation_degrees=pn["max_rotation_degrees"], perturb_pair_transform=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=motordock_se3_collate_fn)
    s = ds[0]; m = cfg["model"]
    model = MotorDockDiffusionModel(s["protein_feat"].shape[-1], s["ligand_atom_feat"].shape[-1], s["pair_features"].shape[-1],
                                    hidden_dim=m["hidden_dim"], num_layers=m["num_layers"], dropout=m.get("dropout",0.1), sigma_emb_dim=m.get("sigma_emb_dim",64),
                                    use_pair_attention=m.get("use_pair_attention", True), adapter_type=m.get("adapter_type","se3"), disable_pair_context=m.get("disable_pair_context", False),
                                    use_motor_auxiliary=m.get("use_motor_auxiliary", True), max_pair_rotation_scale=m.get("max_pair_rotation_scale",0.5), max_pair_translation_scale=m.get("max_pair_translation_scale",5.0))
    model.load_state_dict(ck["model_state_dict"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)

    out_dir = Path(args.out).parent / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for batch in dl:
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = sample_motordock_diffusion(
            model,
            b,
            num_samples=args.num_samples,
            num_steps=cfg["diffusion"]["num_steps"],
            sigma_tr_max=cfg["diffusion"]["sigma_tr_max"],
            sigma_tr_min=cfg["diffusion"]["sigma_tr_min"],
            sigma_rot_max=cfg["diffusion"]["sigma_rot_max"],
            sigma_rot_min=cfg["diffusion"]["sigma_rot_min"],
            sigma_tor_max=cfg["diffusion"].get("sigma_tor_max", 3.14159),
            sigma_tor_min=cfg["diffusion"].get("sigma_tor_min", 0.05),
            ode=cfg["diffusion"].get("ode", False),
            temperature=cfg["diffusion"].get("temperature", 1.0),
            candidate_score_weights=cfg.get("candidate_scoring", None),
        )
        pdb = batch["pdb_id"][0]
        top = out["top_ligand_coords"][0].detach().cpu()
        pth = out_dir / f"{pdb}_top.pt"
        torch.save({"pdb_id": pdb, "top_ligand_coords": top}, pth)
        rows.append({"pdb_id": pdb, "top_sample_index": int(out["top_sample_index"][0].item()), "prediction_path": str(pth)})

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print({"n": len(rows)})
