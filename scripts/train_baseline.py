from __future__ import annotations

import argparse
import yaml
from motordock.train.train_baseline import train_baseline


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = train_baseline(cfg)
    print(out)
