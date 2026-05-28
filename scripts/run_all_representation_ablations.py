from __future__ import annotations
import argparse, copy, os, yaml, subprocess


def merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--representations", nargs="+", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-infer", action="store_true")
    ap.add_argument("--max-train-examples", type=int, default=None)
    ap.add_argument("--max-val-examples", type=int, default=None)
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    base = yaml.safe_load(open(args.base_config, "r", encoding="utf-8"))
    for rep in args.representations:
        p = f"configs/ablations/{rep}.yaml"
        cfg = base
        if os.path.exists(p):
            cfg = merge(base, yaml.safe_load(open(p, "r", encoding="utf-8")))
        cfg["representation"]["name"] = rep
        cfg["output"]["run_dir"] = f"{args.run_root}/{rep}"
        if args.max_train_examples is not None: cfg["data"]["max_train_examples"] = args.max_train_examples
        if args.max_val_examples is not None: cfg["data"]["max_val_examples"] = args.max_val_examples
        tmp = f"/tmp/ablation_{rep}.yaml"
        yaml.safe_dump(cfg, open(tmp, "w", encoding="utf-8"))

        if not args.skip_train:
            subprocess.check_call(["python", "scripts/train_representation_ablation.py", "--config", tmp])
            subprocess.check_call(["python", "scripts/eval_representation_ablation.py", "--checkpoint", f"{args.run_root}/{rep}/best.pt", "--config", tmp, "--split", cfg["data"]["split_val"], "--num-samples", str(args.num_samples), "--out", f"{args.run_root}/{rep}/val_eval.csv"])
        if not args.skip_infer:
            subprocess.check_call(["python", "scripts/infer_representation_ablation.py", "--checkpoint", f"{args.run_root}/{rep}/best.pt", "--config", tmp, "--split", cfg["data"]["split_val"], "--num-samples", str(args.num_samples), "--out", f"{args.run_root}/{rep}/val_predictions.csv"])
