from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motordock.eval.posebusters_runner import run_posebusters_if_available


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--export-dir", default=None)
    ap.add_argument("--config", default="redock", choices=["redock", "dock", "mol"])
    ap.add_argument("--full-report", action="store_true", default=False)
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=0)
    args = ap.parse_args()

    res = run_posebusters_if_available(
        prediction_csv=args.prediction_csv,
        output_csv=args.out,
        export_dir=args.export_dir,
        config=args.config,
        full_report=args.full_report,
        top_n=args.top_n,
        max_workers=args.max_workers,
    )

    print(f"available: {res.get('available', False)}")
    print(f"number of poses evaluated: {res.get('num_poses', 0)}")
    print(f"pb_valid rate: {res.get('pb_valid_rate', 0.0)}")
    print(f"pb_valid_no_rmsd rate: {res.get('pb_valid_no_rmsd_rate', 0.0)}")
    print(f"output path: {res.get('output_csv', args.out)}")
