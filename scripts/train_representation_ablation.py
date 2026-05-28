from __future__ import annotations
import argparse, copy, yaml
from motordock.train.train_representation_ablation import train_representation_ablation

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    print(train_representation_ablation(cfg))
