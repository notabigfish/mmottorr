#!/usr/bin/env bash
set -e
PYTHONPATH=. python scripts/run_all_representation_ablations.py --base-config configs/ablation_debug.yaml --representations se3_log quaternion_translation dual_quaternion matrix centroid_bias random_motor shuffled_pairs no_pair_context  pga_feature pga_sandwich --run-root runs/representation_ablation_debug --max-train-examples 128 --max-val-examples 32 --num-samples 5
PYTHONPATH=. python scripts/summarize_representation_ablations.py --run-root runs/representation_ablation_debug --out runs/representation_ablation_debug/ablation_summary.csv
