from __future__ import annotations

from pathlib import Path
import torch


def save_checkpoint(path: str, ckpt: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(path: str, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location)
