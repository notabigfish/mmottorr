from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from .ablation_metrics import top1_by_confidence
from .statistical_tests import bootstrap_paired_delta_ci


CATS = ["Overall", "Single-Domain", "Intra-Chain-Domain-Interface", "Linker", "Chain-Interface", "Mixed-Domain-Chain"]


def summarize_run(run_root: str) -> pd.DataFrame:
    rows = []
    root = Path(run_root)
    reps = [p.name for p in root.iterdir() if p.is_dir()]
    for rep in reps:
        pred = root / rep / "val_predictions.csv"
        if not pred.exists():
            continue
        df = pd.read_csv(pred)
        top = top1_by_confidence(df)
        nparam = None
        logp = root / rep / "train_log.csv"
        if logp.exists():
            lg = pd.read_csv(logp)
            if "num_parameters" in lg.columns and len(lg) > 0:
                nparam = int(lg["num_parameters"].iloc[-1])
        for c in CATS:
            s = top if c == "Overall" else top[top["motordock_type"] == c]
            if len(s) == 0:
                continue
            rows.append({"representation": rep, "motordock_type": c, "n": len(s), "mean_rmsd": s["rmsd"].mean(), "median_rmsd": s["rmsd"].median(), "success_2A": s["success_2A"].mean(), "mean_centroid_distance": s["centroid_distance"].mean(), "mean_confidence": s["confidence"].mean(), "mean_attention_entropy": s.get("mean_pair_attention_entropy", pd.Series([0]*len(s))).mean(), "num_parameters": nparam, "seconds_per_complex": np.nan})
    out = pd.DataFrame(rows)
    return out


def add_deltas_vs_se3(summary: pd.DataFrame, run_root: str) -> pd.DataFrame:
    root = Path(run_root)
    base = pd.read_csv(root / "se3_log" / "val_predictions.csv")
    base = top1_by_confidence(base)

    extra = []
    for rep in sorted(summary["representation"].unique()):
        pred_path = root / rep / "val_predictions.csv"
        if not pred_path.exists():
            continue
        cur = top1_by_confidence(pd.read_csv(pred_path))
        m = base[["pdb_id", "success_2A", "rmsd", "motordock_type"]].merge(cur[["pdb_id", "success_2A", "rmsd", "motordock_type"]], on="pdb_id", suffixes=("_base", "_rep"))
        for c in CATS:
            s = m if c == "Overall" else m[m["motordock_type_base"] == c]
            if len(s) == 0:
                continue
            ds, lo, hi = bootstrap_paired_delta_ci(s["success_2A_base"].to_numpy().astype(float), s["success_2A_rep"].to_numpy().astype(float), metric="success_rate")
            dr = float(s["rmsd_rep"].mean() - s["rmsd_base"].mean())
            extra.append({"representation": rep, "motordock_type": c, "delta_success_2A_vs_se3": ds, "delta_mean_rmsd_vs_se3": dr, "bootstrap_ci_low": lo, "bootstrap_ci_high": hi})
    ex = pd.DataFrame(extra)
    return summary.merge(ex, on=["representation", "motordock_type"], how="left")
