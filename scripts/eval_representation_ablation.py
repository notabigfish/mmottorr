from __future__ import annotations
import argparse, yaml, pandas as pd, torch
from torch.utils.data import DataLoader
from motordock.train.checkpointing import load_checkpoint
from motordock.train.validate_representation_ablation import validate_representation_multi_sample
from motordock.data.ablation_dataset import RepresentationAblationDataset
from motordock.data.motordock_collate import motordock_se3_collate_fn
from motordock.data.representation_pair_featurizer import representation_pair_feature_dim
from motordock.models.motordock_ablation_model import MotorDockAblationModel, count_trainable_parameters

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
    d,m,r,pn = cfg["data"], cfg["model"], cfg["representation"], cfg["pose_noise"]
    ablation_mode = r["name"] if r["name"] in {"random_motor","shuffled_pairs","no_pair_context"} else "normal"
    ds = RepresentationAblationDataset(d["csv_path"], d["output_dir"], split=args.split, max_examples=d.get("max_val_examples"), require_pocket=d.get("require_pocket",True), max_ligand_atoms=d.get("max_ligand_atoms",128), max_protein_residues=d.get("max_protein_residues",1022), max_candidate_pairs=d.get("max_candidate_pairs",16), randomize_pose=True, max_translation=pn["max_translation"], max_rotation_degrees=pn["max_rotation_degrees"], perturb_pair_transform=False, pair_perturb_prob=0.0, representation=r["name"], matrix_mode=r.get("matrix_mode","3x4"), ablation_mode=ablation_mode, shuffle_pair_features=r.get("shuffle_pair_features",False), random_motor_seed=r.get("random_motor_seed",0))
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"], collate_fn=motordock_se3_collate_fn)
    s = ds[0]
    model = MotorDockAblationModel(s["protein_feat"].shape[-1], s["ligand_atom_feat"].shape[-1], representation_pair_feature_dim(r["name"], r.get("matrix_mode","3x4")), representation=r["name"], matrix_mode=r.get("matrix_mode","3x4"), hidden_dim=m["hidden_dim"], num_layers=m["num_layers"], dropout=m["dropout"], use_pair_attention=m.get("use_pair_attention",True), disable_pair_context=m.get("disable_pair_context",False), parameter_budget_mode=r.get("parameter_budget_mode","matched"), max_rotation_scale=r.get("max_rotation_scale",0.5), max_translation_scale=r.get("max_translation_scale",5.0))
    model.load_state_dict(ck["model_state_dict"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(dev)
    res = validate_representation_multi_sample(model, dl, dev, num_samples=args.num_samples)
    res["representation"] = r["name"]
    res["num_parameters"] = count_trainable_parameters(model)
    pd.DataFrame([res]).to_csv(args.out, index=False)
    print(res)
