from pathlib import Path

import torch

from motordock.geometry.transforms_io import (
    load_domain_frames,
    load_candidate_pairs,
    load_pocket_info,
    build_unit_frame_map,
    attach_pair_transforms,
)
from motordock.geometry.validation import summarize_geometry_for_complex


def test_load_and_validate_fake_geometry_outputs(tmp_path: Path):
    out = tmp_path
    (out / "frames").mkdir(parents=True)
    (out / "candidate_pairs").mkdir(parents=True)
    (out / "pocket_info").mkdir(parents=True)

    pdb_id = "xxxx"
    frames = [
        {
            "unit_id": "A",
            "stable": True,
            "reason": "ok",
            "R": torch.eye(3),
            "t": torch.zeros(3),
            "residue_indices": [0, 1, 2],
        },
        {
            "unit_id": "B",
            "stable": True,
            "reason": "ok",
            "R": torch.eye(3),
            "t": torch.tensor([1.0, 0.0, 0.0]),
            "residue_indices": [3, 4, 5],
        },
    ]
    pairs = [
        {
            "pair_id": "A__B",
            "unit_a": "A",
            "unit_b": "B",
            "pair_type": "domain_pair",
            "has_native_transform": True,
            "T_ab_native": torch.tensor(
                [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            ),
        }
    ]
    pocket = {
        "pocket_selected_indices": torch.tensor([0, 1], dtype=torch.long),
        "pocket_center_ca": torch.zeros(3),
    }

    torch.save(frames, out / "frames" / f"{pdb_id}_domain_frames.pt")
    torch.save(pairs, out / "candidate_pairs" / f"{pdb_id}_candidate_pairs.pt")
    torch.save(pocket, out / "pocket_info" / f"{pdb_id}_pocket_info.pt")

    lf = load_domain_frames(out, pdb_id)
    lp = load_candidate_pairs(out, pdb_id)
    lpi = load_pocket_info(out, pdb_id)

    fmap = build_unit_frame_map(lf)
    attached = attach_pair_transforms(lp, fmap)
    summary = summarize_geometry_for_complex(pdb_id, lf, attached, lpi)

    assert summary["n_frames"] == 2
    assert summary["n_stable_frames"] == 2
    assert summary["n_candidate_pairs"] == 1
    assert summary["n_pairs_with_transform"] == 1
    assert summary["pocket_center_available"] is True
    assert attached[0]["has_recomputed_transform"] is True
