import torch
from motordock.eval.metrics_pose import ligand_rmsd, success_rate


def test_pose_metrics_basic():
    p = torch.zeros(2, 3, 3)
    t = torch.zeros(2, 3, 3)
    m = torch.ones(2, 3, dtype=torch.bool)
    r = ligand_rmsd(p, t, m)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-6)
    assert success_rate(r, 2.0) == 1.0
