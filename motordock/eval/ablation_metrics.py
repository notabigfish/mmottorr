from __future__ import annotations

import pandas as pd


def top1_by_confidence(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["pdb_id", "confidence"], ascending=[True, False]).groupby("pdb_id").head(1).copy()
