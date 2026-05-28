from __future__ import annotations

import csv
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from motordock.data import PDBBindBaselineDataset, baseline_collate_fn
from motordock.models import BaselineDockingModel
from motordock.losses.pose_loss import rigid_docking_loss
from motordock.losses.confidence_loss import confidence_bce_loss
from .validate_baseline import validate_one_sample
from .checkpointing import save_checkpoint


def _seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_baseline(config: dict):
    _seed(config["seed"])
    run_dir = Path(config["output"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    dcfg = config["data"]
    ncfg = config["pose_noise"]
    trcfg = config["train"]

    train_ds = PDBBindBaselineDataset(
        csv_path=dcfg["csv_path"], output_dir=dcfg["output_dir"], split=dcfg["split_train"],
        max_examples=dcfg.get("max_train_examples"), require_pocket=dcfg.get("require_pocket", True),
        max_ligand_atoms=dcfg.get("max_ligand_atoms", 128), max_protein_residues=dcfg.get("max_protein_residues", 1022),
        randomize_pose=True, max_translation=ncfg["max_translation"], max_rotation_degrees=ncfg["max_rotation_degrees"],
    )
    val_ds = PDBBindBaselineDataset(
        csv_path=dcfg["csv_path"], output_dir=dcfg["output_dir"], split=dcfg["split_val"],
        max_examples=dcfg.get("max_val_examples"), require_pocket=dcfg.get("require_pocket", True),
        max_ligand_atoms=dcfg.get("max_ligand_atoms", 128), max_protein_residues=dcfg.get("max_protein_residues", 1022),
        randomize_pose=True, max_translation=ncfg["max_translation"], max_rotation_degrees=ncfg["max_rotation_degrees"],
    )

    train_loader = DataLoader(train_ds, batch_size=trcfg["batch_size"], shuffle=True, num_workers=trcfg["num_workers"], collate_fn=baseline_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=trcfg["batch_size"], shuffle=False, num_workers=trcfg["num_workers"], collate_fn=baseline_collate_fn)

    sample = train_ds[0]
    model = BaselineDockingModel(sample["protein_feat"].shape[-1], sample["ligand_atom_feat"].shape[-1], config["model"]["hidden_dim"], config["model"]["num_layers"], config["model"]["dropout"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=trcfg["lr"], weight_decay=trcfg["weight_decay"])
    use_amp = bool(trcfg.get("use_amp", True) and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    log_path = run_dir / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "step", "loss", "coord", "se3", "rmsd", "lr"])

    best = float("inf")
    step = 0
    for epoch in range(1, trcfg["epochs"] + 1):
        model.train()
        for batch in train_loader:
            b = _to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(b)
                ld = rigid_docking_loss(out, b, trcfg["lambda_coord"], trcfg["lambda_se3"])
                conf_loss = confidence_bce_loss(out["confidence_logit"], ld["rmsd"]) if trcfg.get("lambda_confidence", 0.0) > 0 else torch.tensor(0.0, device=device)
                loss = ld["total"] + trcfg.get("lambda_confidence", 0.0) * conf_loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), trcfg["grad_clip_norm"])
            scaler.step(opt)
            scaler.update()
            step += 1

            if step % trcfg["log_interval"] == 0:
                with open(log_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([epoch, step, float(loss.item()), float(ld["coord_loss"].item()), float(ld["se3_loss"].item()), float(ld["rmsd"].mean().item()), opt.param_groups[0]["lr"]])

        if epoch % trcfg["val_interval"] == 0:
            vm = validate_one_sample(model, val_loader, device, config)
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_amp else None,
                "config": config,
                "best_val_rmsd": best,
            }
            save_checkpoint(str(run_dir / "latest.pt"), ckpt)
            if vm["val_mean_rmsd"] < best:
                best = vm["val_mean_rmsd"]
                ckpt["best_val_rmsd"] = best
                save_checkpoint(str(run_dir / "best.pt"), ckpt)

    return {"best_val_rmsd": best, "run_dir": str(run_dir)}
