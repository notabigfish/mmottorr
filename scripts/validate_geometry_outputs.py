from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from motordock.geometry.se3 import is_valid_rotation, is_valid_transform
from motordock.geometry.transforms_io import (
    load_domain_frames,
    load_candidate_pairs,
    load_pocket_info,
    build_unit_frame_map,
    attach_pair_transforms,
)
from motordock.geometry.validation import summarize_geometry_for_complex
from tqdm import tqdm

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-complexes", type=int, default=100)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--save-report", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    df = pd.read_csv(args.csv)
    if args.split is not None and "split" in df.columns:
        df = df[df["split"] == args.split]
    if args.max_complexes > 0:
        df = df.head(args.max_complexes)

    rows = []
    strict_failed = False
    strict_reason = None

    for _, r in tqdm(df.iterrows(), total=len(df)):
        pdb_id = str(r["Assembly_ID"])
        row = {
            "pdb_id": pdb_id,
            "split": r.get("split", None),
            "motordock_type": r.get("motordock_type", None),
            "status": "ok",
            "reason": "",
            "max_transform_diff_fro": None,
        }
        try:
            frames = load_domain_frames(outdir, pdb_id)
            pairs = load_candidate_pairs(outdir, pdb_id)
            pocket = load_pocket_info(outdir, pdb_id)

            unit_map = build_unit_frame_map(frames)
            pairs_attached = attach_pair_transforms(pairs, unit_map)

            summary = summarize_geometry_for_complex(pdb_id, frames, pairs_attached, pocket)
            row.update(summary)

            diffs = [p.get("transform_diff_fro") for p in pairs_attached if p.get("transform_diff_fro") is not None]
            row["max_transform_diff_fro"] = max(diffs) if diffs else None

            strict_violation = None
            if args.strict:
                if summary["n_invalid_stable_frames"] > 0:
                    strict_violation = "invalid_stable_frame"
                elif summary["n_invalid_pair_transforms"] > 0:
                    strict_violation = "invalid_native_transform"
                elif row["max_transform_diff_fro"] is not None and row["max_transform_diff_fro"] > 1e-3:
                    strict_violation = "transform_diff_gt_1e-3"
                elif not summary["pocket_center_available"]:
                    strict_violation = "missing_pocket_center"

                if strict_violation is None:
                    for fr in frames:
                        if bool(fr.get("stable", False)):
                            R = fr.get("R", None)
                            if R is None or not is_valid_rotation(R).item():
                                strict_violation = "invalid_stable_rotation"
                                break

                if strict_violation is None:
                    for p in pairs:
                        if bool(p.get("has_native_transform", False)):
                            T = p.get("T_ab_native", None)
                            if T is None or (not is_valid_transform(T).item()):
                                strict_violation = "invalid_saved_native_transform"
                                break

            if strict_violation is not None:
                row["status"] = "fail"
                row["reason"] = strict_violation
                strict_failed = True
                strict_reason = f"{pdb_id}:{strict_violation}"

        except FileNotFoundError as e:
            row.update({
                "n_frames": None,
                "n_stable_frames": None,
                "n_invalid_stable_frames": None,
                "max_orthogonality_error": None,
                "min_det": None,
                "max_det": None,
                "n_candidate_pairs": None,
                "n_pairs_with_transform": None,
                "n_invalid_pair_transforms": None,
                "n_pocket_selected_residues": None,
                "pocket_center_available": None,
            })
            row["status"] = "fail" if args.strict else "warn"
            row["reason"] = str(e)
            if args.strict:
                strict_failed = True
                strict_reason = f"{pdb_id}:missing_file"
        except Exception as e:
            row["status"] = "fail" if args.strict else "warn"
            row["reason"] = f"exception:{e}"
            if args.strict:
                strict_failed = True
                strict_reason = f"{pdb_id}:exception"

        rows.append(row)
        if args.strict and strict_failed:
            break

    report = pd.DataFrame(rows)

    summary = {
        "n_total": int(len(report)),
        "n_ok": int((report["status"] == "ok").sum()) if len(report) > 0 else 0,
        "n_warn": int((report["status"] == "warn").sum()) if len(report) > 0 else 0,
        "n_fail": int((report["status"] == "fail").sum()) if len(report) > 0 else 0,
        "strict": bool(args.strict),
        "strict_failed": bool(strict_failed),
        "strict_reason": strict_reason,
    }

    if args.save_report:
        report_path = outdir / "geometry_validation_report.csv"
        summary_path = outdir / "geometry_validation_summary.json"
        report.to_csv(report_path, index=False)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved report: {report_path}")
        print(f"Saved summary: {summary_path}")

    print(json.dumps(summary, indent=2))
    return 1 if (args.strict and strict_failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
