from __future__ import annotations

import torch
from motordock.losses.pose_loss import rigid_docking_loss
from motordock.eval.metrics_pose import ligand_rmsd, centroid_distance, success_rate
from motordock.eval.metrics_confidence import top1_by_confidence
from motordock.diffusion.noise_schedule import DiffusionSchedule
from motordock.models.pose_sampler import DiffusionPoseSampler
from motordock.infer.pose_sampler import OneStepPoseSampler


def _to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def _build_sampler(model, sampler: str, num_samples: int, schedule_cfg: dict | None = None):
    if sampler == "one_step":
        return OneStepPoseSampler(model, num_samples=num_samples)
    schedule_cfg = schedule_cfg or {}
    sched = DiffusionSchedule(
        num_steps=int(schedule_cfg.get("num_steps", 20)),
        sigma_tr_min=float(schedule_cfg.get("sigma_tr_min", 0.1)),
        sigma_tr_max=float(schedule_cfg.get("sigma_tr_max", 10.0)),
        sigma_rot_min=float(schedule_cfg.get("sigma_rot_min", 0.05)),
        sigma_rot_max=float(schedule_cfg.get("sigma_rot_max", 1.5)),
        schedule_type=str(schedule_cfg.get("schedule_type", "log_linear")),
    )
    return DiffusionPoseSampler(
        model,
        sched,
        num_samples=num_samples,
        deterministic=bool(schedule_cfg.get("deterministic", False)),
        center_init=str(schedule_cfg.get("center_init", "pocket")),
        init_translation_sigma=float(schedule_cfg.get("init_translation_sigma", sched.sigma_tr_max)),
        max_step_norm_tr=schedule_cfg.get("max_step_norm_tr", None),
        max_step_norm_rot=schedule_cfg.get("max_step_norm_rot", None),
    )


@torch.no_grad()
def validate_one_sample(model, dataloader, device, config) -> dict:
    model.eval()
    losses, cl, sl, rmsds, cds = [], [], [], [], []
    for batch in dataloader:
        b = _to_device(batch, device)
        out = model(b)
        ld = rigid_docking_loss(out, b, config["train"]["lambda_coord"], config["train"]["lambda_se3"])
        losses.append(float(ld["total"].item()))
        cl.append(float(ld["coord_loss"].item()))
        sl.append(float(ld["se3_loss"].item()))
        r = ligand_rmsd(out["ligand_coords_pred"], b["ligand_coords_true"], b["ligand_mask"])
        c = centroid_distance(out["ligand_coords_pred"], b["ligand_coords_true"], b["ligand_mask"])
        rmsds.append(r.cpu())
        cds.append(c.cpu())

    rmsd = torch.cat(rmsds) if rmsds else torch.zeros(1)
    cd = torch.cat(cds) if cds else torch.zeros(1)
    return {
        "val_loss": float(sum(losses) / max(len(losses), 1)),
        "val_coord_loss": float(sum(cl) / max(len(cl), 1)),
        "val_se3_loss": float(sum(sl) / max(len(sl), 1)),
        "val_mean_rmsd": float(rmsd.mean().item()),
        "val_median_rmsd": float(rmsd.median().item()),
        "val_top1_success_2A": success_rate(rmsd, threshold=2.0),
        "val_centroid_distance": float(cd.mean().item()),
    }


@torch.no_grad()
def validate_multi_sample(
    model,
    dataloader,
    device,
    num_samples: int = 5,
    sampler: str = "one_step",
    schedule_cfg: dict | None = None,
) -> dict:
    model.eval()
    all_r = []
    all_c = []
    pose_sampler = _build_sampler(model, sampler=sampler, num_samples=num_samples, schedule_cfg=schedule_cfg)

    for batch in dataloader:
        b = _to_device(batch, device)
        samp_out = pose_sampler.sample(b)
        coords = samp_out["coords"]
        conf = samp_out["confidence_logit"]

        B, S = coords.shape[:2]
        true = b["ligand_coords_true"].unsqueeze(1).expand(B, S, -1, -1)
        mask = b["ligand_mask"].unsqueeze(1).expand(B, S, -1)
        r = ligand_rmsd(coords.reshape(B * S, *coords.shape[2:]), true.reshape(B * S, *true.shape[2:]), mask.reshape(B * S, -1))
        r = r.view(B, S)

        all_r.append(r.cpu())
        all_c.append(conf.cpu())

    rmsd_mat = torch.cat(all_r, dim=0)
    conf_mat = torch.cat(all_c, dim=0)
    top1 = top1_by_confidence(rmsd_mat, conf_mat)
    oracle = rmsd_mat.min(dim=1).values

    return {
        "top1_by_confidence_rmsd": float(top1.mean().item()),
        "oracle_topk_rmsd": float(oracle.mean().item()),
        "top1_success_2A": float((top1 < 2.0).float().mean().item()),
        "oracle_topk_success_2A": float((oracle < 2.0).float().mean().item()),
    }
