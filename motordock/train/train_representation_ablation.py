from __future__ import annotations

import csv, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from motordock.data.ablation_dataset import RepresentationAblationDataset
from motordock.data.motordock_collate import motordock_se3_collate_fn
from motordock.data.representation_pair_featurizer import representation_pair_feature_dim
from motordock.models.motordock_ablation_model import MotorDockAblationModel, count_trainable_parameters
from motordock.losses.ablation_loss import motordock_ablation_loss
from motordock.eval.metrics_pair import pair_residual_errors, attention_entropy
from motordock.train.validate_representation_ablation import validate_representation_one_sample
from motordock.train.checkpointing import save_checkpoint


def _seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}


def train_representation_ablation(config: dict):
    _seed(config["seed"])
    run_dir = Path(config["output"]["run_dir"]); run_dir.mkdir(parents=True, exist_ok=True)
    d,t,m,r,pn,pp = config["data"], config["train"], config["model"], config["representation"], config["pose_noise"], config["pair_perturbation"]
    ablation_mode = r["name"] if r["name"] in {"random_motor","shuffled_pairs","no_pair_context"} else "normal"

    tr_ds = RepresentationAblationDataset(d["csv_path"], d["output_dir"], split=d["split_train"], max_examples=d.get("max_train_examples"), require_pocket=d.get("require_pocket",True), max_ligand_atoms=d.get("max_ligand_atoms",128), max_protein_residues=d.get("max_protein_residues",1022), max_candidate_pairs=d.get("max_candidate_pairs",16), randomize_pose=True, max_translation=pn["max_translation"], max_rotation_degrees=pn["max_rotation_degrees"], perturb_pair_transform=pp.get("enabled",True), pair_perturb_prob=pp.get("prob",0.5), pair_max_rotation_degrees=pp.get("max_rotation_degrees",10.0), pair_max_translation=pp.get("max_translation",2.0), representation=r["name"], matrix_mode=r.get("matrix_mode","3x4"), ablation_mode=ablation_mode, shuffle_pair_features=r.get("shuffle_pair_features",False), random_motor_seed=r.get("random_motor_seed",0))
    va_ds = RepresentationAblationDataset(d["csv_path"], d["output_dir"], split=d["split_val"], max_examples=d.get("max_val_examples"), require_pocket=d.get("require_pocket",True), max_ligand_atoms=d.get("max_ligand_atoms",128), max_protein_residues=d.get("max_protein_residues",1022), max_candidate_pairs=d.get("max_candidate_pairs",16), randomize_pose=True, max_translation=pn["max_translation"], max_rotation_degrees=pn["max_rotation_degrees"], perturb_pair_transform=False, pair_perturb_prob=0.0, representation=r["name"], matrix_mode=r.get("matrix_mode","3x4"), ablation_mode=ablation_mode, shuffle_pair_features=r.get("shuffle_pair_features",False), random_motor_seed=r.get("random_motor_seed",0))

    tr = DataLoader(tr_ds, batch_size=t["batch_size"], shuffle=True, num_workers=t["num_workers"], collate_fn=motordock_se3_collate_fn)
    va = DataLoader(va_ds, batch_size=t["batch_size"], shuffle=False, num_workers=t["num_workers"], collate_fn=motordock_se3_collate_fn)
    s = tr_ds[0]
    model = MotorDockAblationModel(s["protein_feat"].shape[-1], s["ligand_atom_feat"].shape[-1], representation_pair_feature_dim(r["name"], r.get("matrix_mode","3x4")), representation=r["name"], matrix_mode=r.get("matrix_mode","3x4"), hidden_dim=m["hidden_dim"], num_layers=m["num_layers"], dropout=m["dropout"], use_pair_attention=m.get("use_pair_attention",True), disable_pair_context=m.get("disable_pair_context",False), parameter_budget_mode=r.get("parameter_budget_mode","matched"), max_rotation_scale=r.get("max_rotation_scale",0.5), max_translation_scale=r.get("max_translation_scale",5.0))
    nparam = count_trainable_parameters(model)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    use_amp = bool(t.get("use_amp", True) and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    with open(run_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch","step","representation","num_parameters","total","coord","ligand_se3","pair_motor","confidence","rmsd","pair_rot","pair_trans","attn_entropy","lr","mem_mb"])

    best, step = 1e9, 0
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        for batch in tr:
            b = _to_device(batch, dev)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                o = model(b)
                l = motordock_ablation_loss(o, b, r["name"], t["lambda_coord"], t["lambda_ligand_se3"], t["lambda_pair_motor"], t.get("lambda_confidence",0.0))
            scaler.scale(l["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip_norm"])
            scaler.step(opt); scaler.update(); step += 1
            if step % t["log_interval"] == 0:
                pr, pt = pair_residual_errors(o["pair_delta_T_pred"], b["pair_T_target_residual"], b["pair_mask"])
                ae = attention_entropy(o["pair_attention"], b["pair_mask"])
                mem = float(torch.cuda.max_memory_allocated()/(1024**2)) if torch.cuda.is_available() else 0.0
                with open(run_dir / "train_log.csv", "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([epoch,step,r["name"],nparam,float(l["total"].item()),float(l["coord_loss"].item()),float(l["ligand_se3_loss"].item()),float(l["pair_motor_loss"].item()),float(l["confidence_loss"].item()),float(l["rmsd"].mean().item()),float(pr.item()),float(pt.item()),float(ae.item()),opt.param_groups[0]["lr"],mem])
        if epoch % t["val_interval"] == 0:
            vm = validate_representation_one_sample(model, va, dev, config)
            vm["num_parameters"] = nparam
            ck = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(), "scaler_state_dict": scaler.state_dict() if use_amp else None, "config": config, "best_val_rmsd": best, "model_type": "motordock_ablation", "representation": r["name"], "pair_feat_dim": representation_pair_feature_dim(r["name"], r.get("matrix_mode","3x4"))}
            save_checkpoint(str(run_dir / "latest.pt"), ck)
            if vm["val_mean_rmsd"] < best:
                best = vm["val_mean_rmsd"]; ck["best_val_rmsd"] = best; save_checkpoint(str(run_dir / "best.pt"), ck)
    return {"best_val_rmsd": best, "representation": r["name"], "num_parameters": nparam, "run_dir": str(run_dir)}
