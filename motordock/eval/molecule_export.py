from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch


def _load_pred_coords(row: pd.Series):
    if "pred_coords_path" in row and isinstance(row["pred_coords_path"], str) and row["pred_coords_path"]:
        p = Path(row["pred_coords_path"])
        if p.suffix == ".pt":
            obj = torch.load(p, map_location="cpu")
            if isinstance(obj, dict) and "ligand_coords_pred" in obj:
                return obj["ligand_coords_pred"]
            if torch.is_tensor(obj):
                return obj
            raise ValueError(f"Unsupported pred_coords_path .pt payload: {p}")
        if p.suffix == ".json":
            arr = json.loads(p.read_text(encoding="utf-8"))
            return torch.tensor(arr, dtype=torch.float32)
    if "pred_coords_json" in row and isinstance(row["pred_coords_json"], str) and row["pred_coords_json"]:
        arr = json.loads(row["pred_coords_json"])
        return torch.tensor(arr, dtype=torch.float32)
    raise ValueError("No pred coordinates provided (pred_coords_path or pred_coords_json)")


def export_predicted_ligand_sdf(template_ligand_path, pred_coords, out_sdf_path, sanitize: bool = True):
    from rdkit import Chem
    template_ligand_path = str(template_ligand_path)
    out_sdf_path = str(out_sdf_path)

    if template_ligand_path.lower().endswith(".sdf"):
        sup = Chem.SDMolSupplier(template_ligand_path, sanitize=sanitize, removeHs=False)
        mol = sup[0] if len(sup) > 0 else None
    elif template_ligand_path.lower().endswith(".mol2"):
        mol = Chem.MolFromMol2File(template_ligand_path, sanitize=sanitize, removeHs=False)
    else:
        sdf = Path(template_ligand_path).with_suffix(".sdf")
        mol2 = Path(template_ligand_path).with_suffix(".mol2")
        mol = None
        if sdf.exists():
            sup = Chem.SDMolSupplier(str(sdf), sanitize=sanitize, removeHs=False)
            mol = sup[0] if len(sup) > 0 else None
        if mol is None and mol2.exists():
            mol = Chem.MolFromMol2File(str(mol2), sanitize=sanitize, removeHs=False)

    if mol is None:
        raise ValueError(f"Failed to load template ligand: {template_ligand_path}")

    if not torch.is_tensor(pred_coords):
        pred_coords = torch.tensor(pred_coords, dtype=torch.float32)
    pred_coords = pred_coords.detach().cpu().float()

    if pred_coords.ndim != 2 or pred_coords.shape[1] != 3:
        raise ValueError(f"pred_coords must be [N,3], got {tuple(pred_coords.shape)}")
    if mol.GetNumAtoms() != pred_coords.shape[0]:
        raise ValueError(f"Atom count mismatch template={mol.GetNumAtoms()} pred={pred_coords.shape[0]}")

    if mol.GetNumConformers() == 0:
        conf = Chem.Conformer(mol.GetNumAtoms())
        mol.AddConformer(conf, assignId=True)

    conf = mol.GetConformer(0)
    for i in range(mol.GetNumAtoms()):
        x, y, z = pred_coords[i].tolist()
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))

    Path(out_sdf_path).parent.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(out_sdf_path)
    w.write(mol)
    w.close()
    return out_sdf_path


def prepare_posebusters_table(prediction_csv, export_dir):
    prediction_csv = Path(prediction_csv)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(prediction_csv)

    # normalize required IDs
    if "complex_id" not in df.columns:
        if "pdb_id" in df.columns:
            df["complex_id"] = df["pdb_id"]
        else:
            raise ValueError("prediction_csv must contain complex_id or pdb_id")

    if "rank" not in df.columns:
        if "sample_idx" in df.columns:
            df["rank"] = df["sample_idx"]
        else:
            df["rank"] = 0

    if "protein_path" not in df.columns:
        if "protein_file" in df.columns:
            df["protein_path"] = df["protein_file"]
        else:
            raise ValueError("prediction_csv must contain protein_path/protein_file")

    if "ligand_true_path" not in df.columns:
        if "ligand_file" in df.columns:
            df["ligand_true_path"] = df["ligand_file"]
        else:
            raise ValueError("prediction_csv must contain ligand_true_path/ligand_file")

    if "ligand_template_path" not in df.columns:
        df["ligand_template_path"] = df["ligand_true_path"]

    out_rows = []
    for _, row in df.iterrows():
        complex_id = str(row["complex_id"])
        rank = int(row["rank"])

        if "pred_sdf_path" in row and isinstance(row["pred_sdf_path"], str) and row["pred_sdf_path"]:
            mol_pred = str(row["pred_sdf_path"])
        else:
            pred_coords = _load_pred_coords(row)
            out_sdf = export_dir / f"{complex_id}_rank{rank}.sdf"
            mol_pred = export_predicted_ligand_sdf(
                row["ligand_template_path"],
                pred_coords,
                str(out_sdf),
                sanitize=True,
            )

        out_rows.append(
            {
                "mol_pred": mol_pred,
                "mol_true": str(row["ligand_true_path"]),
                "mol_cond": str(row["protein_path"]),
                "complex_id": complex_id,
                "rank": rank,
            }
        )

    return pd.DataFrame(out_rows)
