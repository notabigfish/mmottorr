from __future__ import annotations

import torch


class DiffusionSchedule:
    def __init__(
        self,
        num_steps: int,
        sigma_tr_min: float,
        sigma_tr_max: float,
        sigma_rot_min: float,
        sigma_rot_max: float,
        sigma_tor_min: float = 0.0314,
        sigma_tor_max: float = 3.1416,
        schedule_type: str = "log_linear",
    ):
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if schedule_type != "log_linear":
            raise ValueError(f"unsupported schedule_type: {schedule_type}")
        self.num_steps = int(num_steps)
        self.sigma_tr_min = float(sigma_tr_min)
        self.sigma_tr_max = float(sigma_tr_max)
        self.sigma_rot_min = float(sigma_rot_min)
        self.sigma_rot_max = float(sigma_rot_max)
        self.sigma_tor_min = float(sigma_tor_min)
        self.sigma_tor_max = float(sigma_tor_max)
        self.schedule_type = schedule_type

    def sigma_tr(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_tr_min * (self.sigma_tr_max / self.sigma_tr_min) ** t

    def sigma_rot(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_rot_min * (self.sigma_rot_max / self.sigma_rot_min) ** t

    def sigma_tor(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_tor_min * (self.sigma_tor_max / self.sigma_tor_min) ** t

    def timesteps(self, device: torch.device | None = None) -> torch.Tensor:
        return torch.linspace(1.0, 0.0, self.num_steps + 1, device=device)

    @staticmethod
    def step_size(t_i: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor:
        return (t_i - t_next).abs()
