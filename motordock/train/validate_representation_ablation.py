from __future__ import annotations

import torch
from motordock.losses.ablation_loss import motordock_ablation_loss
from motordock.eval.metrics_pose import ligand_rmsd, centroid_distance
from motordock.eval.metrics_confidence import top1_by_confidence
from motordock.eval.metrics_pair import pair_residual_errors, attention_entropy


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def validate_representation_one_sample(model, dataloader, device, config):
    model.eval()
    rep = config["representation"]["name"]
    vals = {k: [] for k in ["loss","coord","lig","pair","rmsd","cent","prot","ent"]}
    for batch in dataloader:
        b = _to_device(batch, device)
        o = model(b)
        l = motordock_ablation_loss(o, b, rep, config["train"]["lambda_coord"], config["train"]["lambda_ligand_se3"], config["train"]["lambda_pair_motor"], config["train"].get("lambda_confidence", 0.0))
        vals["loss"].append(float(l["total"].item())); vals["coord"].append(float(l["coord_loss"].item())); vals["lig"].append(float(l["ligand_se3_loss"].item())); vals["pair"].append(float(l["pair_motor_loss"].item()))
        r = ligand_rmsd(o["ligand_coords_pred"], b["ligand_coords_true"], b["ligand_mask"])
        c = centroid_distance(o["ligand_coords_pred"], b["ligand_coords_true"], b["ligand_mask"])
        vals["rmsd"].append(r.cpu()); vals["cent"].append(c.cpu())
        pr, pt = pair_residual_errors(o["pair_delta_T_pred"], b["pair_T_target_residual"], b["pair_mask"])
        vals["prot"].append((float(pr.item()), float(pt.item())))
        vals["ent"].append(float(attention_entropy(o["pair_attention"], b["pair_mask"]).item()))
    rmsd = torch.cat(vals["rmsd"]) if vals["rmsd"] else torch.zeros(1)
    cent = torch.cat(vals["cent"]) if vals["cent"] else torch.zeros(1)
    pro = [x[0] for x in vals["prot"]]; pto = [x[1] for x in vals["prot"]]
    return {
        "val_loss": sum(vals["loss"])/max(1,len(vals["loss"])),
        "val_coord_loss": sum(vals["coord"])/max(1,len(vals["coord"])),
        "val_ligand_se3_loss": sum(vals["lig"])/max(1,len(vals["lig"])),
        "val_pair_motor_loss": sum(vals["pair"])/max(1,len(vals["pair"])),
        "val_mean_rmsd": float(rmsd.mean().item()),
        "val_median_rmsd": float(rmsd.median().item()),
        "val_top1_success_2A": float((rmsd < 2.0).float().mean().item()),
        "val_centroid_distance": float(cent.mean().item()),
        "val_pair_rotation_error": sum(pro)/max(1,len(pro)),
        "val_pair_translation_error": sum(pto)/max(1,len(pto)),
        "val_attention_entropy": sum(vals["ent"])/max(1,len(vals["ent"])),
        "representation": rep,
    }


@torch.no_grad()
def validate_representation_multi_sample(model, dataloader, device, num_samples: int = 5):
    model.eval()
    all_r, all_c = [], []
    for batch in dataloader:
        b = _to_device(batch, device)
        rs, cs = [], []
        for _ in range(num_samples):
            o = model(b)
            rs.append(ligand_rmsd(o["ligand_coords_pred"], b["ligand_coords_true"], b["ligand_mask"]))
            cs.append(o["confidence_logit"])
        all_r.append(torch.stack(rs, dim=1).cpu())
        all_c.append(torch.stack(cs, dim=1).cpu())
    rm = torch.cat(all_r, dim=0)
    cm = torch.cat(all_c, dim=0)
    top1 = top1_by_confidence(rm, cm)
    oracle = rm.min(dim=1).values
    return {
        "top1_by_confidence_rmsd": float(top1.mean().item()),
        "oracle_topk_rmsd": float(oracle.mean().item()),
        "top1_success_2A": float((top1 < 2.0).float().mean().item()),
        "oracle_topk_success_2A": float((oracle < 2.0).float().mean().item()),
    }
