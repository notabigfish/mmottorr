from __future__ import annotations
import argparse, yaml, pandas as pd, torch
from torch.utils.data import DataLoader
from motordock.train.checkpointing import load_checkpoint
from motordock.train.validate_motordock_se3 import validate_motordock_multi_sample
from motordock.data.motordock_dataset import MotorDockSE3Dataset
from motordock.data.motordock_collate import motordock_se3_collate_fn
from motordock.models.motordock_se3_model import MotorDockSE3Model
from motordock.data.pair_featurizer import pair_feature_dim

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    ck = load_checkpoint(args.checkpoint, map_location="cpu")
    d = cfg["data"]; pn = cfg["pose_noise"]
    ds = MotorDockSE3Dataset(d["csv_path"], d["output_dir"], split=args.split, max_examples=d.get("max_val_examples"),
                             require_pocket=d.get("require_pocket", True), max_ligand_atoms=d.get("max_ligand_atoms", 128),
                             max_protein_residues=d.get("max_protein_residues", 1022), max_candidate_pairs=d.get("max_candidate_pairs", 16),
                             randomize_pose=True, max_translation=pn["max_translation"], max_rotation_degrees=pn["max_rotation_degrees"], perturb_pair_transform=False)
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"], collate_fn=motordock_se3_collate_fn)
    s = ds[0]; mcfg = cfg["model"]
    m = MotorDockSE3Model(s["protein_feat"].shape[-1], s["ligand_atom_feat"].shape[-1], pair_feature_dim(), hidden_dim=mcfg["hidden_dim"], num_layers=mcfg["num_layers"], dropout=mcfg["dropout"], use_pair_attention=mcfg.get("use_pair_attention", True), use_motor_auxiliary=mcfg.get("use_motor_auxiliary", True), disable_pair_context=mcfg.get("disable_pair_context", False), freeze_baseline_encoder=mcfg.get("freeze_baseline_encoder", False), max_pair_rotation_scale=mcfg.get("max_pair_rotation_scale", 0.5), max_pair_translation_scale=mcfg.get("max_pair_translation_scale", 5.0))
    m.load_state_dict(ck["model_state_dict"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(dev)
    res = validate_motordock_multi_sample(m, dl, dev, num_samples=args.num_samples)
    pd.DataFrame([res]).to_csv(args.out, index=False)
    print(res)
