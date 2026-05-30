from __future__ import annotations

import torch

from motordock.diffusion.noise_schedule import DiffusionSchedule
from motordock.diffusion.rigid_pose import center_of_mass, apply_rigid_update, random_translation, random_rotation_vec
from motordock.data.pose_noise import sample_random_ligand_transform, apply_transform_to_points


class DiffusionPoseSampler:
    def __init__(
        self,
        model,
        schedule: DiffusionSchedule,
        num_samples: int,
        deterministic: bool = False,
        center_init: str = "pocket",
        init_translation_sigma: float = 10.0,
        max_step_norm_tr: float | None = None,
        max_step_norm_rot: float | None = None,
    ):
        self.model = model
        self.schedule = schedule
        self.num_samples = int(num_samples)
        self.deterministic = bool(deterministic)
        self.center_init = center_init
        self.init_translation_sigma = float(init_translation_sigma)
        self.max_step_norm_tr = max_step_norm_tr
        self.max_step_norm_rot = max_step_norm_rot

    def _repeat_batch(self, batch: dict) -> dict:
        B = batch["ligand_coords_start"].shape[0]
        S = self.num_samples
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.unsqueeze(1).repeat(1, S, *([1] * (v.ndim - 1))).reshape(B * S, *v.shape[1:])
            elif isinstance(v, list):
                out[k] = [x for x in v for _ in range(S)]
            else:
                out[k] = v
        out["_B"] = B
        out["_S"] = S
        return out

    def _clamp_norm(self, x: torch.Tensor, max_norm: float | None) -> torch.Tensor:
        if max_norm is None:
            return x
        n = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
        scale = torch.clamp(torch.tensor(max_norm, device=x.device, dtype=x.dtype) / n, max=1.0)
        return x * scale

    def _init_coords(self, rb: dict) -> torch.Tensor:
        coords = rb["ligand_coords_start"].clone()
        BxS = coords.shape[0]
        ligand_mask = rb["ligand_mask"]
        center = center_of_mass(coords, ligand_mask)

        for i in range(BxS):
            T = sample_random_ligand_transform(
                max_translation=0.0,
                max_rotation_degrees=180.0,
                device=coords.device,
                dtype=coords.dtype,
            )
            c = center[i]
            coords[i] = apply_transform_to_points(T, coords[i] - c) + c

        pocket_center = rb["pocket_center"]
        init_t = random_translation(BxS, self.init_translation_sigma, coords.device).to(coords.dtype)
        if self.center_init == "pocket":
            lig_center = center_of_mass(coords, ligand_mask)
            to_pocket = pocket_center - lig_center
            coords = coords + to_pocket[:, None, :] + init_t[:, None, :]
        else:
            coords = coords + init_t[:, None, :]

        return coords

    @torch.no_grad()
    def sample(self, batch: dict, save_trajectory: bool = False) -> dict:
        self.model.eval()
        rb = self._repeat_batch(batch)
        B, S = rb["_B"], rb["_S"]

        x = self._init_coords(rb)
        traj = [x.clone()] if save_trajectory else None
        ts = self.schedule.timesteps(x.device)

        for i in range(len(ts) - 1):
            t_i = ts[i]
            t_n = ts[i + 1]
            dt = self.schedule.step_size(t_i, t_n)

            sigma_tr = self.schedule.sigma_tr(t_i).expand(x.shape[0]).to(x.dtype)
            sigma_rot = self.schedule.sigma_rot(t_i).expand(x.shape[0]).to(x.dtype)

            step_batch = {k: v for k, v in rb.items() if not k.startswith("_")}
            step_batch["ligand_coords_t"] = x
            step_batch["sigma_tr"] = sigma_tr
            step_batch["sigma_rot"] = sigma_rot
            step_batch["t"] = torch.full((x.shape[0],), float(t_i.item()), device=x.device, dtype=x.dtype)

            out = self.model(step_batch)
            s_tr = out["tr_score_pred"]
            s_rot = out["rot_score_pred"]

            tr_upd = dt * (sigma_tr[:, None] ** 2) * s_tr
            rot_upd = dt * (sigma_rot[:, None] ** 2) * s_rot

            if not self.deterministic:
                z_tr = torch.randn_like(tr_upd)
                z_rot = torch.randn_like(rot_upd)
                tr_upd = tr_upd + torch.sqrt(2.0 * dt) * sigma_tr[:, None] * z_tr
                rot_upd = rot_upd + torch.sqrt(2.0 * dt) * sigma_rot[:, None] * z_rot

            tr_upd = self._clamp_norm(tr_upd, self.max_step_norm_tr)
            rot_upd = self._clamp_norm(rot_upd, self.max_step_norm_rot)

            ctr = center_of_mass(x, rb["ligand_mask"])
            x = apply_rigid_update(x, rot_upd, tr_upd, ctr)
            if save_trajectory:
                traj.append(x.clone())

        final_batch = {k: v for k, v in rb.items() if not k.startswith("_")}
        final_batch["ligand_coords_t"] = x
        final_batch["sigma_tr"] = self.schedule.sigma_tr(torch.tensor(0.0, device=x.device)).expand(x.shape[0]).to(x.dtype)
        final_batch["sigma_rot"] = self.schedule.sigma_rot(torch.tensor(0.0, device=x.device)).expand(x.shape[0]).to(x.dtype)
        conf = self.model(final_batch)["confidence_logit"]

        coords = x.view(B, S, *x.shape[1:])
        conf = conf.view(B, S)
        ranked = torch.argsort(conf, dim=1, descending=True)

        out = {
            "coords": coords,
            "confidence_logit": conf,
            "ranked_indices": ranked,
        }
        if save_trajectory:
            out["trajectory"] = [t.view(B, S, *t.shape[1:]) for t in traj]
        return out
