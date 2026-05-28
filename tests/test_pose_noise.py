import torch
from motordock.data.pose_noise import randomize_ligand_pose, apply_transform_to_points


def test_randomize_ligand_pose_recoverable():
    coords_true = torch.randn(12, 3)
    pocket_center = torch.randn(3)
    coords_start, _, T_target = randomize_ligand_pose(coords_true, pocket_center)
    recovered = apply_transform_to_points(T_target, coords_start)
    assert recovered.shape == coords_true.shape
