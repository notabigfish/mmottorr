from __future__ import annotations

import time
import torch


def num_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def batch_timing(start: float, end: float, batch_size: int) -> dict:
    dt = end - start
    return {"seconds_per_batch": dt, "seconds_per_complex": dt / max(batch_size, 1)}


def peak_cuda_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
