from __future__ import annotations

import torch
from motordock.losses.motordock_loss import motordock_se3_loss
from motordock.eval.metrics_pose import ligand_rmsd, centroid_distance
from motordock.eval.metrics_confidence import top1_by_confidence
from motordock.eval.metrics_pair import pair_residual_errors, attention_entropy


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def validate_motordock_one_sample(model, dataloader, device, config) -> dict:
    model.eval()
    vals = {k: [] for k in ["loss", "coord", "ligse3", "pair", "rmsd", "cent"]}
    pr, pt, ent = [], [], []
    for batch in dataloader:
        b = _to_device(batch, device)
        out = model(b)
        ld = motordock_se3_loss(
            out, b,
            coord_weight=config["train"]["lambda_coord"],
            ligand_se3_weight=config["train"]["lambda_ligand_se3"],
            pair_motor_weight=config["train"]["lambda_pair_motor"],
            confidence_weight=config["train"].get("lambda_confidence", 0.0),
        )
        vals["loss"].append(float(ld["total"].item()))
        vals["coord"].append(float(ld["coord_loss"].item()))
        vals["ligse3"].append(float(ld["ligand_se3_loss"].item()))
        vals["pair"].append(float(ld["pair_motor_loss"].item()))

        r = ligand_rmsd(out["ligand_coords_pred"], b["ligand_coords_true"], b["ligand_mask"])
        c = centroid_distance(out["ligand_coords_pred"], b["ligand_coords_true"], b["ligand_mask"])
        vals["rmsd"].append(r.cpu())
        vals["cent"].append(c.cpu())

        rr, tt = pair_residual_errors(out["pair_delta_T_pred"], b["pair_T_target_residual"], b["pair_mask"])
        pr.append(float(rr.item()))
        pt.append(float(tt.item()))
        ent.append(float(attention_entropy(out["pair_attention"], b["pair_mask"]).item()))

    rmsd = torch.cat(vals["rmsd"]) if vals["rmsd"] else torch.zeros(1)
    cent = torch.cat(vals["cent"]) if vals["cent"] else torch.zeros(1)

    return {
        "val_loss": sum(vals["loss"]) / max(len(vals["loss"]), 1),
        "val_coord_loss": sum(vals["coord"]) / max(len(vals["coord"]), 1),
        "val_ligand_se3_loss": sum(vals["ligse3"]) / max(len(vals["ligse3"]), 1),
        "val_pair_motor_loss": sum(vals["pair"]) / max(len(vals["pair"]), 1),
        "val_mean_rmsd": float(rmsd.mean().item()),
        "val_median_rmsd": float(rmsd.median().item()),
        "val_top1_success_2A": float((rmsd < 2.0).float().mean().item()),
        "val_centroid_distance": float(cent.mean().item()),
        "val_pair_rotation_error": sum(pr) / max(len(pr), 1),
        "val_pair_translation_error": sum(pt) / max(len(pt), 1),
        "val_attention_entropy": sum(ent) / max(len(ent), 1),
    }


@torch.no_grad()
def validate_motordock_multi_sample(model, dataloader, device, num_samples: int = 5) -> dict:
    model.eval()
    all_r, all_c = [], []
    for batch in dataloader:
        b = _to_device(batch, device)
        rs, cs = [], []
        for _ in range(num_samples):
            out = model(b)
            rs.append(ligand_rmsd(out["ligand_coords_pred"], b["ligand_coords_true"], b["ligand_mask"]))
            cs.append(out["confidence_logit"])
        all_r.append(torch.stack(rs, dim=1).cpu())
        all_c.append(torch.stack(cs, dim=1).cpu())
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
