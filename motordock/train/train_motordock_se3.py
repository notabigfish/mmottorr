from __future__ import annotations

import csv
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from motordock.data.motordock_dataset import MotorDockSE3Dataset
from motordock.data.motordock_collate import motordock_se3_collate_fn
from motordock.data.pair_featurizer import pair_feature_dim
from motordock.models.motordock_se3_model import MotorDockSE3Model
from motordock.losses.motordock_loss import motordock_se3_loss
from motordock.eval.metrics_pair import pair_residual_errors, attention_entropy
from .validate_motordock_se3 import validate_motordock_one_sample
from .checkpointing import save_checkpoint


def _seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_motordock_se3(config: dict):
    _seed(config["seed"])
    run_dir = Path(config["output"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    d = config["data"]
    pn = config["pose_noise"]
    pp = config["pair_perturbation"]
    t = config["train"]

    train_ds = MotorDockSE3Dataset(
        d["csv_path"], d["output_dir"], split=d["split_train"], max_examples=d.get("max_train_examples"),
        require_pocket=d.get("require_pocket", True), max_ligand_atoms=d.get("max_ligand_atoms", 128),
        max_protein_residues=d.get("max_protein_residues", 1022), max_candidate_pairs=d.get("max_candidate_pairs", 16),
        randomize_pose=True, max_translation=pn["max_translation"], max_rotation_degrees=pn["max_rotation_degrees"],
        perturb_pair_transform=pp.get("enabled", True), pair_perturb_prob=pp.get("prob", 0.5),
        pair_max_rotation_degrees=pp.get("max_rotation_degrees", 10.0), pair_max_translation=pp.get("max_translation", 2.0),
    )
    val_ds = MotorDockSE3Dataset(
        d["csv_path"], d["output_dir"], split=d["split_val"], max_examples=d.get("max_val_examples"),
        require_pocket=d.get("require_pocket", True), max_ligand_atoms=d.get("max_ligand_atoms", 128),
        max_protein_residues=d.get("max_protein_residues", 1022), max_candidate_pairs=d.get("max_candidate_pairs", 16),
        randomize_pose=True, max_translation=pn["max_translation"], max_rotation_degrees=pn["max_rotation_degrees"],
        perturb_pair_transform=False, pair_perturb_prob=0.0,
    )

    tr = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, num_workers=t["num_workers"], collate_fn=motordock_se3_collate_fn)
    va = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False, num_workers=t["num_workers"], collate_fn=motordock_se3_collate_fn)

    s = train_ds[0]
    mcfg = config["model"]
    model = MotorDockSE3Model(
        s["protein_feat"].shape[-1], s["ligand_atom_feat"].shape[-1], pair_feature_dim(),
        hidden_dim=mcfg["hidden_dim"], num_layers=mcfg["num_layers"], dropout=mcfg["dropout"],
        use_pair_attention=mcfg.get("use_pair_attention", True), use_motor_auxiliary=mcfg.get("use_motor_auxiliary", True),
        disable_pair_context=mcfg.get("disable_pair_context", False), freeze_baseline_encoder=mcfg.get("freeze_baseline_encoder", False),
        max_pair_rotation_scale=mcfg.get("max_pair_rotation_scale", 0.5), max_pair_translation_scale=mcfg.get("max_pair_translation_scale", 5.0),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    use_amp = bool(t.get("use_amp", True) and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    log_path = run_dir / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "step", "total", "coord", "ligse3", "pair", "conf", "rmsd", "pair_rot", "pair_trans", "attn_ent", "lr", "mem_mb"])

    best = 1e9
    step = 0
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        for batch in tr:
            b = _to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(b)
                ld = motordock_se3_loss(
                    out, b,
                    coord_weight=t["lambda_coord"],
                    ligand_se3_weight=t["lambda_ligand_se3"],
                    pair_motor_weight=t["lambda_pair_motor"],
                    confidence_weight=t.get("lambda_confidence", 0.0),
                )
            scaler.scale(ld["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip_norm"])
            scaler.step(opt)
            scaler.update()
            step += 1

            if step % t["log_interval"] == 0:
                pr, pt = pair_residual_errors(out["pair_delta_T_pred"], b["pair_T_target_residual"], b["pair_mask"])
                ae = attention_entropy(out["pair_attention"], b["pair_mask"])
                mem = float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
                with open(log_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([epoch, step, float(ld["total"].item()), float(ld["coord_loss"].item()), float(ld["ligand_se3_loss"].item()), float(ld["pair_motor_loss"].item()), float(ld["confidence_loss"].item()), float(ld["rmsd"].mean().item()), float(pr.item()), float(pt.item()), float(ae.item()), opt.param_groups[0]["lr"], mem])

        if epoch % t["val_interval"] == 0:
            vm = validate_motordock_one_sample(model, va, device, config)
            ck = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_amp else None,
                "config": config,
                "best_val_rmsd": best,
                "model_type": "motordock_se3",
                "pair_feat_dim": pair_feature_dim(),
            }
            save_checkpoint(str(run_dir / "latest.pt"), ck)
            if vm["val_mean_rmsd"] < best:
                best = vm["val_mean_rmsd"]
                ck["best_val_rmsd"] = best
                save_checkpoint(str(run_dir / "best.pt"), ck)

    return {"best_val_rmsd": best, "run_dir": str(run_dir)}
