#!/usr/bin/env bash
set -e
python scripts/train_baseline.py --config configs/baseline_debug.yaml
python scripts/eval_baseline.py --checkpoint runs/baseline_debug/best.pt --config configs/baseline_debug.yaml --split val --num-samples 5 --out runs/baseline_debug/val_eval.csv
