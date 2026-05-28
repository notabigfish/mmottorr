#!/usr/bin/env bash
set -e
python scripts/train_motordock_se3.py --config configs/motordock_se3_debug.yaml
python scripts/eval_motordock_se3.py --checkpoint runs/motordock_se3_debug/best.pt --config configs/motordock_se3_debug.yaml --split val --num-samples 5 --out runs/motordock_se3_debug/val_eval.csv
python scripts/infer_motordock_se3.py --checkpoint runs/motordock_se3_debug/best.pt --csv data/pdbbind/pdbbind_ft.csv --output-dir data/pdbbind --split val --num-samples 5 --out runs/motordock_se3_debug/val_predictions.csv
