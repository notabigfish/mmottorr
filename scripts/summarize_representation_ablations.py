from __future__ import annotations
import argparse
from motordock.eval.ablation_report import summarize_run, add_deltas_vs_se3

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    s = summarize_run(args.run_root)
    s = add_deltas_vs_se3(s, args.run_root)
    s.to_csv(args.out, index=False)
    print(s)
