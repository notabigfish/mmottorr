from __future__ import annotations
import argparse
from motordock.eval.compare_baseline_motordock import compare_predictions

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-csv", required=True)
    ap.add_argument("--motordock-csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = compare_predictions(args.baseline_csv, args.motordock_csv)
    df.to_csv(args.out, index=False)
    print(df)
