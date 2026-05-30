from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from motordock.data import PDBBindBaselineDataset, baseline_collate_fn
from motordock.diffusion.noise_schedule import DiffusionSchedule
from motordock.diffusion.rigid_pose import prepare_diffusion_batch_targets
from motordock.losses.pose_loss import diffusion_rigid_loss
from motordock.models import DiffusionDockingModel
from .checkpointing import save_checkpoint
from .validate_diffusion_baseline import validate_diffusion_loss, validate_diffusion_sampling


def _seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_diffusion_baseline(config: dict):
    _seed(config["seed"])
    run_dir = Path(config["output"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    dcfg = config["data"]
    ncfg = config["pose_noise"]
    mcfg = config["model"]
    trcfg = config["train"]
    tdcfg = config["train_diffusion"]
    diffcfg = config["diffusion"]

    train_ds = PDBBindBaselineDataset(
        csv_path=dcfg["csv_path"],
        output_dir=dcfg["output_dir"],
        split=dcfg["split_train"],
        max_examples=dcfg.get("max_train_examples"),
        require_pocket=dcfg.get("require_pocket", True),
        max_ligand_atoms=dcfg.get("max_ligand_atoms", 128),
        max_protein_residues=dcfg.get("max_protein_residues", 1022),
        randomize_pose=True,
        max_translation=ncfg["max_translation"],
        max_rotation_degrees=ncfg["max_rotation_degrees"],
    )
    val_ds = PDBBindBaselineDataset(
        csv_path=dcfg["csv_path"],
        output_dir=dcfg["output_dir"],
        split=dcfg["split_val"],
        max_examples=dcfg.get("max_val_examples"),
        require_pocket=dcfg.get("require_pocket", True),
        max_ligand_atoms=dcfg.get("max_ligand_atoms", 128),
        max_protein_residues=dcfg.get("max_protein_residues", 1022),
        randomize_pose=True,
        max_translation=ncfg["max_translation"],
        max_rotation_degrees=ncfg["max_rotation_degrees"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=trcfg["batch_size"],
        shuffle=True,
        num_workers=trcfg["num_workers"],
        collate_fn=baseline_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=trcfg["batch_size"],
        shuffle=False,
        num_workers=trcfg["num_workers"],
        collate_fn=baseline_collate_fn,
    )

    sample = train_ds[0]
    model = DiffusionDockingModel(
        sample["protein_feat"].shape[-1],
        sample["ligand_atom_feat"].shape[-1],
        mcfg["hidden_dim"],
        mcfg["num_layers"],
        mcfg.get("dropout", 0.1),
    )

    schedule = DiffusionSchedule(
        num_steps=int(diffcfg["num_steps"]),
        sigma_tr_min=float(diffcfg["sigma_tr_min"]),
        sigma_tr_max=float(diffcfg["sigma_tr_max"]),
        sigma_rot_min=float(diffcfg["sigma_rot_min"]),
        sigma_rot_max=float(diffcfg["sigma_rot_max"]),
        schedule_type=str(diffcfg.get("schedule_type", "log_linear")),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=trcfg["lr"], weight_decay=trcfg["weight_decay"])
    use_amp = bool(trcfg.get("use_amp", True) and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    log_path = run_dir / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch",
            "step",
            "loss",
            "tr_loss",
            "rot_loss",
            "tor_loss",
            "lr",
        ])

    best = float("inf")
    step = 0

    for epoch in range(1, trcfg["epochs"] + 1):
        model.train()
        for batch in train_loader:
            b = _to_device(batch, device)
            B = b["protein_feat"].shape[0]
            t = torch.rand(B, device=device, dtype=b["protein_feat"].dtype)
            sigma_tr = schedule.sigma_tr(t)
            sigma_rot = schedule.sigma_rot(t)
            bt = prepare_diffusion_batch_targets(b, sigma_tr, sigma_rot)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(bt)
                ld = diffusion_rigid_loss(
                    out,
                    bt,
                    bt["sigma_tr"],
                    bt["sigma_rot"],
                    lambda_tr=float(tdcfg.get("lambda_tr", 1.0)),
                    lambda_rot=float(tdcfg.get("lambda_rot", 1.0)),
                )
                loss = ld["total"]

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), trcfg["grad_clip_norm"])
            scaler.step(opt)
            scaler.update()
            step += 1

            if step % trcfg["log_interval"] == 0:
                with open(log_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        epoch,
                        step,
                        float(loss.item()),
                        float(ld["tr_loss"].item()),
                        float(ld["rot_loss"].item()),
                        opt.param_groups[0]["lr"],
                    ])

        if epoch % trcfg["val_interval"] == 0:
            val_mode = str(tdcfg.get("val_mode", "dual")).lower()

            vm_loss = validate_diffusion_loss(model, val_loader, device, config)
            vm_sampling = {}
            if val_mode in {"dual", "sampling"}:
                vm_sampling = validate_diffusion_sampling(
                    model,
                    val_loader,
                    device,
                    config,
                    num_samples=int(tdcfg.get("sampling_val_num_samples", 5)),
                )

            if val_mode in {"dual", "sampling"} and "val_top1_by_confidence_rmsd" in vm_sampling:
                current_metric = float(vm_sampling["val_top1_by_confidence_rmsd"])
            else:
                current_metric = float(vm_loss["val_diffusion_loss"])

            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_amp else None,
                "config": config,
                "model_type": "diffusion_baseline",
                "best_val_metric": best,
            }
            save_checkpoint(str(run_dir / "latest.pt"), ckpt)
            if current_metric < best:
                best = current_metric
                ckpt["best_val_metric"] = best
                save_checkpoint(str(run_dir / "best.pt"), ckpt)

    return {
        "best_val_metric": best,
        "run_dir": str(run_dir),
    }
