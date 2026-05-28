from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_delta_success_ci(
    baseline_success: np.ndarray,
    motordock_success: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(baseline_success)
    if n == 0:
        return (0.0, 0.0)
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        deltas.append(float(motordock_success[idx].mean() - baseline_success[idx].mean()))
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def compare_predictions(baseline_csv: str, motordock_csv: str) -> pd.DataFrame:
    b = pd.read_csv(baseline_csv)
    m = pd.read_csv(motordock_csv)

    b1 = b.sort_values(["pdb_id", "confidence"], ascending=[True, False]).groupby("pdb_id").head(1)
    m1 = m.sort_values(["pdb_id", "confidence"], ascending=[True, False]).groupby("pdb_id").head(1)
    z = b1.merge(m1, on="pdb_id", suffixes=("_baseline", "_motordock"))

    cats = [
        "Overall",
        "Single-Domain",
        "Intra-Chain-Domain-Interface",
        "Linker",
        "Chain-Interface",
        "Mixed-Domain-Chain",
    ]
    rows = []
    for c in cats:
        s = z if c == "Overall" else z[z["motordock_type_baseline"] == c]
        if len(s) == 0:
            continue
        bs = s["success_2A_baseline"].to_numpy().astype(float)
        ms = s["success_2A_motordock"].to_numpy().astype(float)
        lo, hi = bootstrap_delta_success_ci(bs, ms)
        rows.append({
            "category": c,
            "n": len(s),
            "mean_rmsd_baseline": s["rmsd_baseline"].mean(),
            "mean_rmsd_motordock": s["rmsd_motordock"].mean(),
            "median_rmsd_baseline": s["rmsd_baseline"].median(),
            "median_rmsd_motordock": s["rmsd_motordock"].median(),
            "success_2A_baseline": bs.mean(),
            "success_2A_motordock": ms.mean(),
            "delta_success_2A": ms.mean() - bs.mean(),
            "delta_success_2A_ci_lo": lo,
            "delta_success_2A_ci_hi": hi,
            "mean_centroid_baseline": s["centroid_distance_baseline"].mean(),
            "mean_centroid_motordock": s["centroid_distance_motordock"].mean(),
        })
    return pd.DataFrame(rows)
