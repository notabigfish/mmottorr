from __future__ import annotations

from pathlib import Path
import torch

from .frames import frame_from_saved_dict, compute_pair_transform_from_frames


def load_domain_frames(output_dir: str | Path, pdb_id: str) -> list[dict]:
    """Load frames/{pdb_id}_domain_frames.pt."""
    path = Path(output_dir) / "frames" / f"{pdb_id}_domain_frames.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing frames file: {path}")
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, list):
        raise TypeError(f"Expected list for frames: {path}")
    return data


def load_candidate_pairs(output_dir: str | Path, pdb_id: str) -> list[dict]:
    """Load candidate_pairs/{pdb_id}_candidate_pairs.pt."""
    path = Path(output_dir) / "candidate_pairs" / f"{pdb_id}_candidate_pairs.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing candidate_pairs file: {path}")
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, list):
        raise TypeError(f"Expected list for candidate_pairs: {path}")
    return data


def load_pocket_info(output_dir: str | Path, pdb_id: str) -> dict:
    """Load pocket_info/{pdb_id}_pocket_info.pt."""
    path = Path(output_dir) / "pocket_info" / f"{pdb_id}_pocket_info.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing pocket_info file: {path}")
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict for pocket_info: {path}")
    return data


def build_unit_frame_map(frames: list[dict]) -> dict[str, torch.Tensor]:
    """Return unit_id -> 4x4 frame for stable frames only."""
    out: dict[str, torch.Tensor] = {}
    for fr in frames:
        unit_id = fr.get("unit_id", None)
        if unit_id is None:
            continue
        T = frame_from_saved_dict(fr)
        if T is not None:
            out[str(unit_id)] = T
    return out


def attach_pair_transforms(
    candidate_pairs: list[dict],
    unit_frame_map: dict[str, torch.Tensor],
) -> list[dict]:
    """Recompute T_ab for each pair and attach diagnostics."""
    out = []
    for pair in candidate_pairs:
        p = dict(pair)
        ua = p.get("unit_a", None)
        ub = p.get("unit_b", None)
        has = ua in unit_frame_map and ub in unit_frame_map
        p["has_recomputed_transform"] = bool(has)
        p["transform_diff_fro"] = None
        if has:
            T_rec = compute_pair_transform_from_frames(unit_frame_map[ua], unit_frame_map[ub])
            p["T_ab_recomputed"] = T_rec
            T_native = p.get("T_ab_native", None)
            if T_native is not None:
                if not torch.is_tensor(T_native):
                    T_native = torch.tensor(T_native, dtype=T_rec.dtype)
                T_native = T_native.to(dtype=T_rec.dtype)
                if T_native.shape == (4, 4):
                    p["transform_diff_fro"] = float(torch.linalg.norm(T_native - T_rec, ord="fro"))
        out.append(p)
    return out
