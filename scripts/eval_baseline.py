from __future__ import annotations

import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from motordock.train.checkpointing import load_checkpoint
from motordock.data import PDBBindBaselineDataset, baseline_collate_fn
from motordock.models import BaselineDockingModel, DiffusionDockingModel
from motordock.train.validate_baseline import validate_multi_sample


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--sampler", choices=["diffusion", "one_step"], default="diffusion")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ck = load_checkpoint(args.checkpoint, map_location="cpu")

    ds = PDBBindBaselineDataset(
        csv_path=cfg["data"]["csv_path"], output_dir=cfg["data"]["output_dir"], split=args.split,
        max_examples=cfg["data"].get("max_val_examples"), require_pocket=cfg["data"].get("require_pocket", True),
        max_ligand_atoms=cfg["data"].get("max_ligand_atoms", 128), max_protein_residues=cfg["data"].get("max_protein_residues", 1022),
        randomize_pose=True, max_translation=cfg["pose_noise"]["max_translation"], max_rotation_degrees=cfg["pose_noise"]["max_rotation_degrees"],
    )
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"], collate_fn=baseline_collate_fn)

    sample = ds[0]
    if args.sampler == "diffusion":
        m = DiffusionDockingModel(sample["protein_feat"].shape[-1], sample["ligand_atom_feat"].shape[-1], cfg["model"]["hidden_dim"], cfg["model"]["num_layers"], cfg["model"].get("dropout", 0.1))
    else:
        m = BaselineDockingModel(sample["protein_feat"].shape[-1], sample["ligand_atom_feat"].shape[-1], cfg["model"]["hidden_dim"], cfg["model"]["num_layers"], cfg["model"]["dropout"])

    try:
        m.load_state_dict(ck["model_state_dict"])
    except Exception as e:
        raise RuntimeError(f"checkpoint incompatible with sampler={args.sampler}: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(device)

    res = validate_multi_sample(
        m,
        dl,
        device,
        num_samples=args.num_samples,
        sampler=args.sampler,
        schedule_cfg=cfg.get("diffusion", None),
    )
    import pandas as pd
    pd.DataFrame([res]).to_csv(args.out, index=False)
    print(res)
