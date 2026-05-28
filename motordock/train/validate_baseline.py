from __future__ import annotations

import torch
from motordock.losses.pose_loss import rigid_docking_loss
from motordock.eval.metrics_pose import ligand_rmsd, centroid_distance, success_rate
from motordock.eval.metrics_confidence import top1_by_confidence


def _to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


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
def validate_multi_sample(model, dataloader, device, num_samples: int = 5) -> dict:
    model.eval()
    all_r = []
    all_c = []
    for batch in dataloader:
        b = _to_device(batch, device)
        rs, cs = [], []
        for _ in range(num_samples):
            out = model(b)
            r = ligand_rmsd(out["ligand_coords_pred"], b["ligand_coords_true"], b["ligand_mask"])
            rs.append(r)
            cs.append(out["confidence_logit"])
        rmat = torch.stack(rs, dim=1)
        cmat = torch.stack(cs, dim=1)
        all_r.append(rmat.cpu())
        all_c.append(cmat.cpu())

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
