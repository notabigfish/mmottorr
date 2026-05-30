from __future__ import annotations

import torch

from motordock.diffusion.noise_schedule import DiffusionSchedule
from motordock.diffusion.rigid_pose import prepare_diffusion_batch_targets
from motordock.losses.pose_loss import diffusion_rigid_loss
from motordock.models.pose_sampler import DiffusionPoseSampler
from motordock.eval.metrics_pose import ligand_rmsd
from motordock.eval.metrics_confidence import top1_by_confidence


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _sample_t(B: int, device, dtype=torch.float32) -> torch.Tensor:
    return torch.rand(B, device=device, dtype=dtype)


def _sigmas(schedule: DiffusionSchedule, t: torch.Tensor):
    return schedule.sigma_tr(t), schedule.sigma_rot(t)


@torch.no_grad()
def validate_diffusion_loss(model, dataloader, device, config: dict) -> dict:
    model.eval()
    dcfg = config["diffusion"]
    tcfg = config["train_diffusion"]
    schedule = DiffusionSchedule(
        num_steps=int(dcfg["num_steps"]),
        sigma_tr_min=float(dcfg["sigma_tr_min"]),
        sigma_tr_max=float(dcfg["sigma_tr_max"]),
        sigma_rot_min=float(dcfg["sigma_rot_min"]),
        sigma_rot_max=float(dcfg["sigma_rot_max"]),
        schedule_type=str(dcfg.get("schedule_type", "log_linear")),
    )

    losses = []
    tr_losses = []
    rot_losses = []
    tor_losses = []
    for batch in dataloader:
        b = _to_device(batch, device)
        B = b["protein_feat"].shape[0]
        t = _sample_t(B, device=device, dtype=b["protein_feat"].dtype)
        sigma_tr, sigma_rot = _sigmas(schedule, t)
        sigma_tor = schedule.sigma_tor(t)
        bt = prepare_diffusion_batch_targets(b, sigma_tr, sigma_rot)
        bt["sigma_tor"] = sigma_tor

        out = model(bt)
        ld = diffusion_rigid_loss(
            out,
            bt,
            bt["sigma_tr"],
            bt["sigma_rot"],
            sigma_tor=bt.get("sigma_tor", None),
            lambda_tr=float(tcfg.get("lambda_tr", 1.0)),
            lambda_rot=float(tcfg.get("lambda_rot", 1.0)),
            lambda_tor=float(tcfg.get("lambda_tor", 1.0)),
        )
        losses.append(float(ld["total"].item()))
        tr_losses.append(float(ld["tr_loss"].item()))
        rot_losses.append(float(ld["rot_loss"].item()))
        tor_losses.append(float(ld.get("tor_loss", ld["total"] * 0.0).item()))

    return {
        "val_diffusion_loss": float(sum(losses) / max(len(losses), 1)),
        "val_tr_loss": float(sum(tr_losses) / max(len(tr_losses), 1)),
        "val_rot_loss": float(sum(rot_losses) / max(len(rot_losses), 1)),
        "val_tor_loss": float(sum(tor_losses) / max(len(tor_losses), 1)),
    }


@torch.no_grad()
def validate_diffusion_sampling(model, dataloader, device, config: dict, num_samples: int | None = None) -> dict:
    model.eval()
    dcfg = config["diffusion"]
    tcfg = config["train_diffusion"]
    S = int(num_samples if num_samples is not None else tcfg.get("sampling_val_num_samples", 5))

    schedule = DiffusionSchedule(
        num_steps=int(dcfg["num_steps"]),
        sigma_tr_min=float(dcfg["sigma_tr_min"]),
        sigma_tr_max=float(dcfg["sigma_tr_max"]),
        sigma_rot_min=float(dcfg["sigma_rot_min"]),
        sigma_rot_max=float(dcfg["sigma_rot_max"]),
        schedule_type=str(dcfg.get("schedule_type", "log_linear")),
    )
    sampler = DiffusionPoseSampler(
        model=model,
        schedule=schedule,
        num_samples=S,
        deterministic=bool(dcfg.get("deterministic_eval", True)),
        center_init=str(dcfg.get("center_init", "pocket")),
        init_translation_sigma=float(dcfg.get("init_translation_sigma", dcfg["sigma_tr_max"])),
        max_step_norm_tr=dcfg.get("max_step_norm_tr", None),
        max_step_norm_rot=dcfg.get("max_step_norm_rot", None),
    )

    all_r = []
    all_c = []
    for batch in dataloader:
        b = _to_device(batch, device)
        samp = sampler.sample(b)
        coords = samp["coords"]  # [B,S,N,3]
        conf = samp["confidence_logit"]  # [B,S]

        B = coords.shape[0]
        true = b["ligand_coords_true"].unsqueeze(1).expand(B, S, -1, -1)
        mask = b["ligand_mask"].unsqueeze(1).expand(B, S, -1)
        rmsd = ligand_rmsd(
            coords.reshape(B * S, *coords.shape[2:]),
            true.reshape(B * S, *true.shape[2:]),
            mask.reshape(B * S, -1),
        ).view(B, S)

        all_r.append(rmsd.cpu())
        all_c.append(conf.cpu())

    rmsd_mat = torch.cat(all_r, dim=0)
    conf_mat = torch.cat(all_c, dim=0)
    top1 = top1_by_confidence(rmsd_mat, conf_mat)
    oracle = rmsd_mat.min(dim=1).values

    return {
        "val_top1_by_confidence_rmsd": float(top1.mean().item()),
        "val_oracle_topk_rmsd": float(oracle.mean().item()),
        "val_top1_success_2A": float((top1 < 2.0).float().mean().item()),
        "val_oracle_topk_success_2A": float((oracle < 2.0).float().mean().item()),
    }
