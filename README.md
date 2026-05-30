# MotorDock

## Geometry

1. SE(3) exp/log tests pass.
2. Frame construction produces valid right-handed frames.
3. No holo contacts are used for candidate selection.
4. Motor residual loss is numerically stable.
```bash
PYTHONPATH=. python scripts/validate_geometry_outputs.py \
  --csv data/pdbbind/pdbbind_ft.csv \
  --output-dir data/pdbbind \
  --max-complexes 100 \
  --save-report
```
```bash
PYTHONPATH=. python scripts/validate_geometry_outputs.py \
  --csv data/pdbbind/pdbbind_ft.csv \
  --output-dir data/pdbbind \
  --max-complexes 100 \
  --strict \
  --save-report
```

## Baseline 
1. Base docking model trains.
2. Validation loss decreases.
3. Top-1 RMSD and Top-5 RMSD can be computed.
4. PoseBusters runner works.

```bash
PYTHONPATH=. python scripts/train_baseline.py --config configs/baseline_debug.yaml
PYTHONPATH=. python scripts/eval_baseline.py --checkpoint runs/baseline_debug/best.pt --config configs/baseline_debug.yaml --split val --num-samples 5 --out runs/baseline_debug/val_eval.csv
PYTHONPATH=. python scripts/infer_baseline.py --checkpoint runs/baseline_debug/best.pt --csv data/pdbbind/pdbbind_ft.csv --output-dir data/pdbbind --split test --num-samples 5 --out runs/baseline_debug/test_predictions.csv
```


## MotorDock-SE(3)
1. Adapter trains without memory failure.
2. Interface/linker subset improves over base backbone.
3. Single-domain subset does not show artificial large gain.

```bash
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
```


## Representation ablation
1. Quaternion, dual-quaternion, matrix, SE(3), and PGA variants run under matched settings.
2. PGA is not claimed unless it outperforms simpler variants.

PGA variants:
- `pga_feature`: passive PGA motor feature baseline only. It uses motor coefficients as features and does not apply sandwich action.
- `pga_sandwich`: true PGA/Clifford motor adapter. It applies sandwich group action on geometric primitives inside the network.

Claims about Clifford/PGA-specific benefits must come from `pga_sandwich`, not from `pga_feature`.

```bash
bash scripts/run_ablation_debug.sh
```
