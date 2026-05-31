from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from motordock.data.motordock_dataset import MotorDockSE3Dataset
from motordock.data.motordock_collate import motordock_se3_collate_fn
from motordock.diffusion.noise_schedule import DiffusionSchedule
from motordock.diffusion.rigid_pose import prepare_diffusion_batch_targets
from motordock.losses.pose_loss import diffusion_rigid_loss
from motordock.models import MotorDockDiffusionModel
from motordock.train.checkpointing import save_checkpoint
from motordock.train.validate_diffusion_baseline import _to_device


def _seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main(cfg: dict):
    _seed(cfg["seed"])
    run_dir = Path(cfg["output"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    d = cfg["data"]
    pn = cfg["pose_noise"]
    m = cfg["model"]
    tr = cfg["train"]
    diff = cfg["diffusion"]
    ls = cfg.get("loss", {})

    train_ds = MotorDockSE3Dataset(
        d["csv_path"], d["output_dir"], split=d["split_train"], max_examples=d.get("max_train_examples"),
        require_pocket=d.get("require_pocket", True), max_ligand_atoms=d.get("max_ligand_atoms", 128),
        max_protein_residues=d.get("max_protein_residues", 1022), max_candidate_pairs=d.get("max_candidate_pairs", 16),
        randomize_pose=True, max_translation=pn["max_translation"], max_rotation_degrees=pn["max_rotation_degrees"],
        perturb_pair_transform=True,
    )
    val_ds = MotorDockSE3Dataset(
        d["csv_path"], d["output_dir"], split=d["split_val"], max_examples=d.get("max_val_examples"),
        require_pocket=d.get("require_pocket", True), max_ligand_atoms=d.get("max_ligand_atoms", 128),
        max_protein_residues=d.get("max_protein_residues", 1022), max_candidate_pairs=d.get("max_candidate_pairs", 16),
        randomize_pose=True, max_translation=pn["max_translation"], max_rotation_degrees=pn["max_rotation_degrees"],
        perturb_pair_transform=False,
    )

    train_dl = DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=True, num_workers=tr["num_workers"], collate_fn=motordock_se3_collate_fn)
    val_dl = DataLoader(val_ds, batch_size=tr["batch_size"], shuffle=False, num_workers=tr["num_workers"], collate_fn=motordock_se3_collate_fn)

    s = train_ds[0]
    model = MotorDockDiffusionModel(
        protein_feat_dim=s["protein_feat"].shape[-1],
        ligand_feat_dim=s["ligand_atom_feat"].shape[-1],
        pair_feat_dim=s["pair_features"].shape[-1],
        hidden_dim=m.get("hidden_dim", 256),
        num_layers=m.get("num_layers", 4),
        dropout=m.get("dropout", 0.1),
        sigma_emb_dim=m.get("sigma_emb_dim", 64),
        use_pair_attention=m.get("use_pair_attention", True),
        adapter_type=m.get("adapter_type", "se3"),
        disable_pair_context=m.get("disable_pair_context", False),
        use_motor_auxiliary=m.get("use_motor_auxiliary", True),
        max_pair_rotation_scale=m.get("max_pair_rotation_scale", 0.5),
        max_pair_translation_scale=m.get("max_pair_translation_scale", 5.0),
    )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    use_amp = bool(tr.get("use_amp", True) and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    sched = DiffusionSchedule(
        num_steps=int(diff["num_steps"]),
        sigma_tr_min=float(diff["sigma_tr_min"]),
        sigma_tr_max=float(diff["sigma_tr_max"]),
        sigma_rot_min=float(diff["sigma_rot_min"]),
        sigma_rot_max=float(diff["sigma_rot_max"]),
        sigma_tor_min=float(diff.get("sigma_tor_min", 0.05)),
        sigma_tor_max=float(diff.get("sigma_tor_max", 3.14159)),
    )

    logp = run_dir / "train_log.csv"
    with open(logp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "step", "loss", "tr", "rot", "tor", "lr"])

    best = float("inf")
    step = 0
    for epoch in range(1, tr["epochs"] + 1):
        model.train()
        for batch in train_dl:
            b = _to_device(batch, dev)
            B = b["protein_feat"].shape[0]
            t = torch.rand(B, device=dev, dtype=b["protein_feat"].dtype)
            sigma_tr = sched.sigma_tr(t)
            sigma_rot = sched.sigma_rot(t)
            sigma_tor = sched.sigma_tor(t)
            bt = prepare_diffusion_batch_targets(b, sigma_tr, sigma_rot)
            bt["sigma_tor"] = sigma_tor

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(bt)
                ld = diffusion_rigid_loss(
                    out, bt, bt["sigma_tr"], bt["sigma_rot"], sigma_tor=bt["sigma_tor"],
                    lambda_tr=float(ls.get("lambda_tr", 1.0)),
                    lambda_rot=float(ls.get("lambda_rot", 1.0)),
                    lambda_tor=float(ls.get("lambda_tor", 1.0)),
                )
                loss = ld["total"]

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip_norm"])
            scaler.step(opt)
            scaler.update()
            step += 1

            if step % tr["log_interval"] == 0:
                with open(logp, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([epoch, step, float(loss.item()), float(ld["tr_loss"].item()), float(ld["rot_loss"].item()), float(ld["tor_loss"].item()), opt.param_groups[0]["lr"]])

        # lightweight val metric: diffusion loss only
        if epoch % tr["val_interval"] == 0:
            model.eval()
            vals = []
            with torch.no_grad():
                for batch in val_dl:
                    b = _to_device(batch, dev)
                    B = b["protein_feat"].shape[0]
                    t = torch.rand(B, device=dev, dtype=b["protein_feat"].dtype)
                    sigma_tr = sched.sigma_tr(t)
                    sigma_rot = sched.sigma_rot(t)
                    sigma_tor = sched.sigma_tor(t)
                    bt = prepare_diffusion_batch_targets(b, sigma_tr, sigma_rot)
                    bt["sigma_tor"] = sigma_tor
                    out = model(bt)
                    ld = diffusion_rigid_loss(out, bt, bt["sigma_tr"], bt["sigma_rot"], sigma_tor=bt["sigma_tor"],
                                             lambda_tr=float(ls.get("lambda_tr", 1.0)), lambda_rot=float(ls.get("lambda_rot", 1.0)), lambda_tor=float(ls.get("lambda_tor", 1.0)))
                    vals.append(float(ld["total"].item()))
            v = sum(vals) / max(len(vals), 1)

            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_amp else None,
                "config": cfg,
                "best_val_metric": best,
                "model_type": "motordock_diffusion",
            }
            save_checkpoint(str(run_dir / "latest.pt"), ckpt)
            if v < best:
                best = v
                ckpt["best_val_metric"] = best
                save_checkpoint(str(run_dir / "best.pt"), ckpt)

    return {"best_val_metric": best, "run_dir": str(run_dir)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(main(cfg))
