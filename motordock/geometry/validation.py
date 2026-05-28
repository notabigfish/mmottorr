from __future__ import annotations

import torch

from .se3 import is_valid_rotation, is_valid_transform


def validate_frame_dict(frame: dict, atol: float = 1e-4) -> dict:
    """Validate one saved frame dict."""
    unit_id = frame.get("unit_id")
    stable = bool(frame.get("stable", False))
    reason = frame.get("reason", "")
    R = frame.get("R", None)
    t = frame.get("t", None)
    has_R = R is not None
    has_t = t is not None

    det = None
    orthogonality_error = None
    valid_rotation = False
    valid_translation = False
    if has_R:
        if not torch.is_tensor(R):
            R = torch.tensor(R)
        if R.shape == (3, 3):
            det = float(torch.det(R))
            I = torch.eye(3, dtype=R.dtype, device=R.device)
            orthogonality_error = float(torch.linalg.norm(R.T @ R - I, ord="fro"))
            valid_rotation = bool(is_valid_rotation(R, atol=atol).item())

    if has_t:
        if not torch.is_tensor(t):
            t = torch.tensor(t)
        valid_translation = bool(t.shape == (3,) and torch.isfinite(t).all().item())

    num_residues = len(frame.get("residue_indices", []))

    return {
        "unit_id": unit_id,
        "stable": stable,
        "reason": reason,
        "has_R": has_R,
        "has_t": has_t,
        "det": det,
        "orthogonality_error": orthogonality_error,
        "valid_rotation": valid_rotation,
        "valid_translation": valid_translation,
        "num_residues": num_residues,
    }


def validate_candidate_pair(pair: dict, atol: float = 1e-4) -> dict:
    """Validate one candidate pair dictionary."""
    pair_id = pair.get("pair_id", None)
    has_unit_a = pair.get("unit_a", None) is not None
    has_unit_b = pair.get("unit_b", None) is not None
    has_pair_type = pair.get("pair_type", None) is not None

    has_native = bool(pair.get("has_native_transform", False))
    T_native = pair.get("T_ab_native", None)
    native_shape_ok = False
    native_valid = False

    if has_native and T_native is not None:
        if not torch.is_tensor(T_native):
            T_native = torch.tensor(T_native)
        native_shape_ok = T_native.shape == (4, 4)
        if native_shape_ok:
            native_valid = bool(is_valid_transform(T_native, atol=atol).item())

    return {
        "pair_id": pair_id,
        "has_pair_id": pair_id is not None,
        "has_unit_a": has_unit_a,
        "has_unit_b": has_unit_b,
        "has_pair_type": has_pair_type,
        "has_native_transform": has_native,
        "native_shape_ok": native_shape_ok,
        "native_valid": native_valid,
        "is_valid": (pair_id is not None and has_unit_a and has_unit_b and has_pair_type and ((not has_native) or (native_shape_ok and native_valid))),
    }


def summarize_geometry_for_complex(
    pdb_id: str,
    frames: list[dict],
    candidate_pairs: list[dict],
    pocket_info: dict | None = None,
) -> dict:
    """Summarize geometry quality for one complex."""
    fvals = [validate_frame_dict(f) for f in frames]
    stable_frames = [v for v in fvals if v["stable"]]
    invalid_stable = [v for v in stable_frames if not (v["valid_rotation"] and v["valid_translation"])]

    ortho_vals = [v["orthogonality_error"] for v in stable_frames if v["orthogonality_error"] is not None]
    det_vals = [v["det"] for v in stable_frames if v["det"] is not None]

    pvals = [validate_candidate_pair(p) for p in candidate_pairs]
    with_transform = [p for p in pvals if p["has_native_transform"]]
    invalid_pair_transforms = [p for p in with_transform if not (p["native_shape_ok"] and p["native_valid"])]

    n_pocket_selected = None
    pocket_center_available = False
    if pocket_info is not None:
        pidx = pocket_info.get("pocket_selected_indices", None)
        if pidx is not None:
            n_pocket_selected = int(len(pidx))
        pocket_center_available = (pocket_info.get("pocket_center_ca", None) is not None) or (
            pocket_info.get("pocket_center_atom", None) is not None
        )

    return {
        "pdb_id": pdb_id,
        "n_frames": len(frames),
        "n_stable_frames": len(stable_frames),
        "n_invalid_stable_frames": len(invalid_stable),
        "max_orthogonality_error": max(ortho_vals) if ortho_vals else None,
        "min_det": min(det_vals) if det_vals else None,
        "max_det": max(det_vals) if det_vals else None,
        "n_candidate_pairs": len(candidate_pairs),
        "n_pairs_with_transform": len(with_transform),
        "n_invalid_pair_transforms": len(invalid_pair_transforms),
        "n_pocket_selected_residues": n_pocket_selected,
        "pocket_center_available": pocket_center_available,
    }
