from __future__ import annotations
import argparse, yaml
from motordock.train.train_motordock_se3 import train_motordock_se3

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    print(train_motordock_se3(cfg))
