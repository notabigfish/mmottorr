from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .molecule_export import prepare_posebusters_table


_LOAD_META_BOOL_COLS = {
    "mol_pred_loaded",
    "mol_true_loaded",
    "mol_cond_loaded",
}


def _pb_valid_columns(df: pd.DataFrame):
    bool_cols = [c for c in df.columns if df[c].dtype == bool]
    chem_cols = [c for c in bool_cols if c not in _LOAD_META_BOOL_COLS]
    rmsd_cols = [c for c in chem_cols if "rmsd" in c.lower()]
    no_rmsd = [c for c in chem_cols if c not in rmsd_cols]
    return chem_cols, no_rmsd


def run_posebusters_if_available(
    prediction_csv: str,
    output_csv: str,
    export_dir: Optional[str] = None,
    config: str = "redock",
    full_report: bool = True,
    top_n: Optional[int] = None,
    max_workers: int = 0,
) -> dict:
    try:
        from posebusters import PoseBusters
    except Exception:
        return {
            "available": False,
            "reason": "posebusters not installed",
            "prediction_csv": prediction_csv,
            "output_csv": output_csv,
        }

    output_csv = str(output_csv)
    export_dir = str(export_dir or (Path(output_csv).parent / "posebusters_sdf"))

    table = prepare_posebusters_table(prediction_csv, export_dir)
    buster = PoseBusters(config=config, top_n=top_n, max_workers=max_workers)
    report = buster.bust_table(table[["mol_pred", "mol_true", "mol_cond"]], full_report=full_report)

    report = report.reset_index(drop=True)
    report = pd.concat([table[["complex_id", "rank"]].reset_index(drop=True), report], axis=1)

    cols_all, cols_no_rmsd = _pb_valid_columns(report)
    if cols_all:
        report["pb_valid"] = report[cols_all].all(axis=1)
    else:
        report["pb_valid"] = False

    if cols_no_rmsd:
        report["pb_valid_no_rmsd"] = report[cols_no_rmsd].all(axis=1)
    else:
        report["pb_valid_no_rmsd"] = report["pb_valid"]

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_csv, index=False)

    return {
        "available": True,
        "prediction_csv": prediction_csv,
        "output_csv": output_csv,
        "export_dir": export_dir,
        "num_poses": int(len(report)),
        "pb_valid_rate": float(report["pb_valid"].mean()) if len(report) else 0.0,
        "pb_valid_no_rmsd_rate": float(report["pb_valid_no_rmsd"].mean()) if len(report) else 0.0,
        "config": config,
    }
