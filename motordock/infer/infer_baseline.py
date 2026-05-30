from __future__ import annotations

from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from motordock.data import PDBBindBaselineDataset, baseline_collate_fn
from motordock.models import BaselineDockingModel, DiffusionDockingModel
from motordock.models.pose_sampler import DiffusionPoseSampler
from motordock.infer.pose_sampler import OneStepPoseSampler
from motordock.diffusion.noise_schedule import DiffusionSchedule
from motordock.eval.metrics_pose import ligand_rmsd, centroid_distance
from motordock.train.checkpointing import load_checkpoint


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _build_model(ck: dict, sample: dict, sampler: str):
    cfg = ck["config"]
    if sampler == "diffusion":
        model = DiffusionDockingModel(
            sample["protein_feat"].shape[-1],
            sample["ligand_atom_feat"].shape[-1],
            cfg["model"]["hidden_dim"],
            cfg["model"]["num_layers"],
            cfg["model"].get("dropout", 0.1),
        )
    else:
        model = BaselineDockingModel(
            sample["protein_feat"].shape[-1],
            sample["ligand_atom_feat"].shape[-1],
            cfg["model"]["hidden_dim"],
            cfg["model"]["num_layers"],
            cfg["model"]["dropout"],
        )
    try:
        model.load_state_dict(ck["model_state_dict"])
    except Exception as e:
        raise RuntimeError(f"checkpoint incompatible with sampler={sampler}: {e}")
    return model


def run_inference(
    checkpoint: str,
    csv_path: str,
    output_dir: str,
    split: str,
    num_samples: int,
    out_csv: str,
    sampler: str = "diffusion",
):
    ck = load_checkpoint(checkpoint, map_location="cpu")
    cfg = ck["config"]
    dcfg = cfg["data"]
    ncfg = cfg["pose_noise"]

    ds = PDBBindBaselineDataset(
        csv_path=csv_path,
        output_dir=output_dir,
        split=split,
        max_examples=dcfg.get("max_val_examples"),
        require_pocket=dcfg.get("require_pocket", True),
        max_ligand_atoms=dcfg.get("max_ligand_atoms", 128),
        max_protein_residues=dcfg.get("max_protein_residues", 1022),
        randomize_pose=True,
        max_translation=ncfg["max_translation"],
        max_rotation_degrees=ncfg["max_rotation_degrees"],
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=baseline_collate_fn)

    sample = ds[0]
    model = _build_model(ck, sample, sampler=sampler)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if sampler == "diffusion":
        dcfg_diff = cfg.get("diffusion", {})
        schedule = DiffusionSchedule(
            num_steps=int(dcfg_diff.get("num_steps", 20)),
            sigma_tr_min=float(dcfg_diff.get("sigma_tr_min", 0.1)),
            sigma_tr_max=float(dcfg_diff.get("sigma_tr_max", ncfg.get("max_translation", 10.0))),
            sigma_rot_min=float(dcfg_diff.get("sigma_rot_min", 0.05)),
            sigma_rot_max=float(dcfg_diff.get("sigma_rot_max", 1.5)),
            schedule_type=str(dcfg_diff.get("schedule_type", "log_linear")),
        )
        pose_sampler = DiffusionPoseSampler(
            model,
            schedule,
            num_samples=num_samples,
            deterministic=bool(dcfg_diff.get("deterministic", False)),
            center_init=str(dcfg_diff.get("center_init", "pocket")),
            init_translation_sigma=float(dcfg_diff.get("init_translation_sigma", schedule.sigma_tr_max)),
            max_step_norm_tr=dcfg_diff.get("max_step_norm_tr", None),
            max_step_norm_rot=dcfg_diff.get("max_step_norm_rot", None),
        )
    else:
        pose_sampler = OneStepPoseSampler(model, num_samples=num_samples)

    pred_dir = Path(out_csv).parent / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.no_grad():
        for batch in dl:
            b = _to_device(batch, device)
            samp = pose_sampler.sample(b)
            coords = samp["coords"]  # [1,S,N,3]
            conf = samp["confidence_logit"][0]
            pdb_id = batch["pdb_id"][0]

            S = coords.shape[1]
            true = b["ligand_coords_true"].unsqueeze(1).expand(1, S, -1, -1)
            mask = b["ligand_mask"].unsqueeze(1).expand(1, S, -1)
            r = ligand_rmsd(
                coords.reshape(S, *coords.shape[2:]),
                true.reshape(S, *true.shape[2:]),
                mask.reshape(S, -1),
            )
            c = centroid_distance(
                coords.reshape(S, *coords.shape[2:]),
                true.reshape(S, *true.shape[2:]),
                mask.reshape(S, -1),
            )

            for sidx in range(S):
                ppath = pred_dir / f"{pdb_id}_sample{sidx}.pt"
                torch.save(
                    {
                        "pdb_id": pdb_id,
                        "ligand_coords_pred": coords[0, sidx].detach().cpu(),
                        "ligand_coords_true": b["ligand_coords_true"][0].detach().cpu(),
                        "ligand_coords_start": b["ligand_coords_start"][0].detach().cpu(),
                        "confidence": float(conf[sidx].item()),
                        "rmsd": float(r[sidx].item()),
                    },
                    ppath,
                )
                rows.append(
                    {
                        "pdb_id": pdb_id,
                        "split": batch["split"][0],
                        "motordock_type": batch["motordock_type"][0],
                        "sample_idx": sidx,
                        "confidence": float(conf[sidx].item()),
                        "rmsd": float(r[sidx].item()),
                        "centroid_distance": float(c[sidx].item()),
                        "success_2A": int(r[sidx].item() < 2.0),
                        "protein_file": batch["protein_file"][0],
                        "ligand_file": batch["ligand_file"][0],
                        "pocket_file": batch["pocket_file"][0],
                        "prediction_path": str(ppath),
                    }
                )

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv
