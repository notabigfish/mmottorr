from __future__ import annotations
import argparse, yaml
from motordock.infer.infer_representation_ablation import run_inference

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    print(run_inference(args.checkpoint, cfg, args.split, args.num_samples, args.out))
