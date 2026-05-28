from __future__ import annotations

import numpy as np


def bootstrap_paired_delta_ci(x: np.ndarray, y: np.ndarray, metric: str = "success_rate", n_bootstrap: int = 1000, seed: int = 0, ci: float = 0.95) -> tuple[float, float, float]:
    assert len(x) == len(y)
    if len(x) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = len(x)
    if metric == "success_rate":
        delta = float(y.mean() - x.mean())
    elif metric == "mean_rmsd":
        delta = float(y.mean() - x.mean())
    else:
        raise ValueError(metric)
    vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        vals.append(float(y[idx].mean() - x[idx].mean()))
    a = (1.0 - ci) / 2.0
    return delta, float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))
