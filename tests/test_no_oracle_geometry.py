import torch

from motordock.geometry.transforms_io import build_unit_frame_map, attach_pair_transforms


def test_no_oracle_geometry_without_ligand_data():
    frames = [
        {
            "unit_id": "U1",
            "stable": True,
            "R": torch.eye(3),
            "t": torch.zeros(3),
            "residue_indices": [0, 1],
        },
        {
            "unit_id": "U2",
            "stable": True,
            "R": torch.eye(3),
            "t": torch.tensor([0.0, 1.0, 0.0]),
            "residue_indices": [2, 3],
        },
    ]
    pairs = [
        {
            "pair_id": "U1__U2",
            "unit_a": "U1",
            "unit_b": "U2",
            "pair_type": "mixed_pair",
            "has_native_transform": False,
        }
    ]

    fmap = build_unit_frame_map(frames)
    out = attach_pair_transforms(pairs, fmap)

    assert out[0]["has_recomputed_transform"] is True
    assert "T_ab_recomputed" in out[0]
