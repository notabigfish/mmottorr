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

## Change log
### Diffusion pose sampler
- Added iterative ligand rigid-pose diffusion over translation and rotation.
- Added score-matching training targets.
- Replaced toy pose sampler behavior that returned `range(num_samples)`.
- Added CPU tests for sampler shape, finite training targets, denoising behavior, and gradient flow.

### Diffusion baseline training
- Added dedicated end-to-end diffusion trainer: `scripts/train_diffusion_baseline.py`.
- Adds diffusion-native validation with score loss and sampler-based RMSD metrics.
- Produces diffusion checkpoints (`model_type: diffusion_baseline`) for direct `--sampler diffusion` eval/infer.

### MotorDock diffusion
```bash
PYTHONPATH=. python scripts/train_motordock_diffusion.py --config configs/motordock_diffusion_pdbbind.yaml
PYTHONPATH=. python scripts/eval_motordock_diffusion.py --checkpoint runs/motordock_diffusion_pdbbind/best.pt --config configs/motordock_diffusion_pdbbind.yaml --split val --num-samples 20 --out runs/motordock_diffusion_pdbbind/val_eval.csv
```

### Ligand torsion modeling
- Added RDKit-based rotatable bond detection.
- Added differentiable torsion coordinate updates.
- Added torsion diffusion noise, score target, loss, and sampler update.
- Added tests for butane, benzene, angle wrapping, padding masks, and sampler integration.

### PoseBusters execution
- Replaced PoseBusters stub with real molecule export and PoseBusters API execution.
- Added predicted ligand SDF export from template ligand plus predicted coordinates.
- Added pb_valid and pb_valid_no_rmsd summary columns.
- Added CLI script for PoseBusters evaluation.
- Added tests for SDF export, input table creation, unavailable PoseBusters behavior, and mocked PoseBusters reporting.

### Random-motor ablation fix
- Replaced learned RandomMotorAdapter placeholder with a frozen random SE(3) ablation.
- Added deterministic per-complex random motors using stable hashing.
- Added valid SE(3) generation through se3_exp_map.
- Added tests for reproducibility, zero trainable parameters, valid transforms, and no-gradient behavior.
