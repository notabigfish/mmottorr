from __future__ import annotations

import argparse
from motordock.infer.infer_baseline import run_inference


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--sampler", choices=["diffusion", "one_step"], default="diffusion")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print(run_inference(args.checkpoint, args.csv, args.output_dir, args.split, args.num_samples, args.out, sampler=args.sampler))
