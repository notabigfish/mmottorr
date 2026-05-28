cd /rds/homes/s/sxz325/shuo/7788/motordock
PYTHONPATH=. python scripts/validate_geometry_outputs.py \
  --csv data/pdbbind/pdbbind_ft.csv \
  --output-dir data/pdbbind \
  --max-complexes 100 \
  --save-report

PYTHONPATH=. python scripts/validate_geometry_outputs.py \
  --csv data/pdbbind/pdbbind_ft.csv \
  --output-dir data/pdbbind \
  --max-complexes 100 \
  --strict \
  --save-report

Milestone 3
cd /rds/homes/s/sxz325/shuo/7788/motordock
PYTHONPATH=. python scripts/train_baseline.py --config configs/baseline_debug.yaml
PYTHONPATH=. python scripts/eval_baseline.py --checkpoint runs/baseline_debug/best.pt --config configs/baseline_debug.yaml --split val --num-samples 5 --out runs/baseline_debug/val_eval.csv
PYTHONPATH=. python scripts/infer_baseline.py --checkpoint runs/baseline_debug/best.pt --csv data/pdbbind/pdbbind_ft.csv --output-dir data/pdbbind --split test --num-samples 5 --out runs/baseline_debug/test_predictions.csv

Milestone 4

PYTHONPATH=. python scripts/train_motordock_se3.py --config configs/motordock_se3_debug.yaml

PYTHONPATH=. python scripts/eval_motordock_se3.py \
  --checkpoint runs/motordock_se3_debug/best.pt \
  --config configs/motordock_se3_debug.yaml \
  --split val \
  --num-samples 5 \
  --out runs/motordock_se3_debug/val_eval.csv

PYTHONPATH=. python scripts/infer_motordock_se3.py \
  --checkpoint runs/motordock_se3_debug/best.pt \
  --csv data/pdbbind/pdbbind_ft.csv \
  --output-dir data/pdbbind \
  --split test \
  --num-samples 5 \
  --out runs/motordock_se3_debug/test_predictions.csv

PYTHONPATH=. python scripts/compare_baseline_motordock.py \
  --baseline-csv runs/baseline_debug/test_predictions.csv \
  --motordock-csv runs/motordock_se3_debug/test_predictions.csv \
  --out runs/motordock_se3_debug/baseline_vs_motordock.csv