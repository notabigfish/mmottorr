# python pretrain_data_pipeline.py --task gen_motordock_pretrain --end_idx 1000 --allow_single_domain_candidates
import json
import os
import sys
from argparse import ArgumentParser
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBIO, PDBParser, Select
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from scipy.spatial import distance
from tqdm import tqdm

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

COMMON_AAS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


parser = ArgumentParser()
parser.add_argument("--task", type=str, required=True, choices=["gen_pretrain", "validate_pretrain"], help="Task to perform")
parser.add_argument("--start_idx", type=int, default=0, help="Start index for processing")
parser.add_argument("--end_idx", type=int, default=-1, help="End index for processing")
parser.add_argument("--pfam_lst_path", type=str, default="/rds/homes/s/sxz325/shuo/7788/multi-domain-enzyme/data/pdb_pfam_mapping_0517.tsv", help="Path to PFAM list file")
parser.add_argument("--biolip_df_path", type=str, default="/rds/homes/s/sxz325/shuo/7788/multi-domain-enzyme-pro/data/qbiolip/01_qbiolip.csv", help="Path to BioLiP dataframe CSV file")
parser.add_argument("--biolip_pdb_root", type=str, default="/rds/homes/s/sxz325/shuo/7788/multi-domain-enzyme-pro/data/qbiolip/nonredund_rec", help="Path to BioLiP PDB root directory")
parser.add_argument("--biolip_lig_root", type=str, default="/rds/homes/s/sxz325/shuo/7788/multi-domain-enzyme-pro/data/qbiolip/nonredund_lig", help="Path to BioLiP ligand root directory")
parser.add_argument("--max_seq_len", type=int, default=1022, help="Maximum sequence length for processing")

parser.add_argument("--data_root", type=str, default="/rds/homes/s/sxz325/shuo/7788/motordock/data/qbiolip")
parser.add_argument("--motordock_csv_name", type=str, default="01_qbiolip.csv")
parser.add_argument("--contact_cutoff", type=float, default=6.0)
parser.add_argument("--pocket_radius", type=float, default=15.0)
parser.add_argument("--frame_radius", type=float, default=12.0)
parser.add_argument("--max_domain_pairs", type=int, default=8)
parser.add_argument("--min_frame_residues", type=int, default=8)
parser.add_argument("--frame_bootstrap_iters", type=int, default=20)
parser.add_argument("--frame_eigen_ratio_min", type=float, default=0.15)
parser.add_argument("--save_json_debug", action="store_true")
parser.add_argument("--allow_single_domain_candidates", action="store_true")
parser.add_argument("--num_workers", type=int, default=-1)

args = parser.parse_args()


class ResSelect(Select):
    def __init__(self, keep_set):
        self.keep_set = keep_set

    def accept_residue(self, residue):
        return (residue.get_parent().get_id(), residue.get_id()) in self.keep_set


def safe_id(x):
    return str(x).replace("/", "_").replace("\\", "_").replace(" ", "_")


def resolve_num_workers(num_workers):
    if num_workers == -1:
        total = os.cpu_count() or 1
        return max(1, total - 2)
    return max(1, int(num_workers))


def start_end_suffix(args):
    return f"_{args.start_idx}_{args.end_idx}" if args.end_idx != -1 else ""


def format_pfam_domains(group):
    rows = [[
        row["PFAM_ACCESSION"],
        int(row["AUTH_PDBRES_START"]),
        int(row["AUTH_PDBRES_END"]),
        row["CHAIN"],
    ] for _, row in group.iterrows()]
    return pd.DataFrame(rows, columns=["pfam_id", "start", "end", "chain_id"])


def build_domain_lookup(pfam_subset):
    domains = format_pfam_domains(pfam_subset)
    domain_lookup = {}
    for d in domains.itertuples():
        domain_lookup.setdefault(d.chain_id, []).append((d.start, d.end, d.pfam_id))
    return domain_lookup


def extract_standard_residues(model, domain_lookup, common_aas):
    all_residues = []
    gidx = 0
    for chain in model:
        c_id = chain.get_id()
        current_chain_domains = domain_lookup.get(c_id, [])
        for residue in chain:
            if "CA" not in residue:
                continue
            resname = residue.get_resname().strip().upper()
            if resname not in common_aas:
                continue
            res_num = int(residue.get_id()[1])
            icode = residue.get_id()[2].strip() or ""
            assigned_domain = "Linker"
            cid_domain = f"{c_id}_Linker"
            for d_start, d_end, d_id in current_chain_domains:
                if d_start <= res_num <= d_end:
                    assigned_domain = str(d_id)
                    cid_domain = f"{c_id}_{d_id}"
                    break
            all_residues.append({
                "global_idx": gidx,
                "chain": c_id,
                "resid": res_num,
                "icode": icode,
                "resname": resname,
                "coord": residue["CA"].get_coord().astype(np.float32),
                "pfam_domain": assigned_domain,
                "cid_domain": cid_domain,
                "obj": residue,
            })
            gidx += 1
    return all_residues


def save_subset_pdb(model, indices, all_residues, filename):
    keep_set = set()
    for idx in indices:
        res_meta = all_residues[idx]
        keep_set.add((res_meta["chain"], res_meta["obj"].get_id()))
    io = PDBIO()
    io.set_structure(model)
    io.save(filename, select=ResSelect(keep_set))


def compute_ligand_features(lig_file):
    lig = Chem.MolFromPDBFile(lig_file, sanitize=False, removeHs=False)
    if lig is None or lig.GetNumConformers() == 0:
        return None
    conf = lig.GetConformer()
    atom_coord = np.array(conf.GetPositions(), dtype=np.float32)
    atom_symbol = [a.GetSymbol() for a in lig.GetAtoms()]
    if atom_coord.shape[0] == 0:
        return None
    ligand_center = atom_coord.mean(axis=0).astype(np.float32)
    return {
        "mol": lig,
        "atom_coord": atom_coord,
        "atom_symbol": atom_symbol,
        "ligand_center": ligand_center,
        "ligand_atom_count": int(atom_coord.shape[0]),
    }


def compute_contact_and_pocket_masks(prot_ca_coords, lig_coords, args):
    dmat = distance.cdist(prot_ca_coords, lig_coords, "euclidean")
    dists_to_ligand = dmat.min(axis=1)
    contact_residue_mask = dists_to_ligand <= float(args.contact_cutoff)
    known_pocket_center = lig_coords.mean(axis=0).astype(np.float32)
    d_to_center = np.linalg.norm(prot_ca_coords - known_pocket_center[None, :], axis=1)
    pocket_residue_mask = d_to_center <= float(args.pocket_radius)
    return dists_to_ligand, contact_residue_mask, known_pocket_center, pocket_residue_mask


def build_units(all_residues, include_chain_units=True):
    units = []
    residue_domain_id = [None] * len(all_residues)
    residue_unit_id = [None] * len(all_residues)
    domain_mask = [0] * len(all_residues)

    pfam_groups = {}
    chain_groups = {}
    for i, r in enumerate(all_residues):
        chain_groups.setdefault(r["chain"], []).append(i)
        if r["pfam_domain"] != "Linker":
            key = (r["chain"], r["pfam_domain"])
            pfam_groups.setdefault(key, []).append(i)

    for (chain, pfam_id), idxs in sorted(pfam_groups.items(), key=lambda x: (x[0][0], x[0][1])):
        resid_vals = [all_residues[j]["resid"] for j in idxs]
        unit_id = f"chain:{chain}|pfam:{pfam_id}|range:{min(resid_vals)}-{max(resid_vals)}"
        units.append({
            "unit_id": unit_id,
            "unit_type": "pfam_domain",
            "chain_id": chain,
            "pfam_id": pfam_id,
            "residue_indices": idxs,
        })

    for chain, idxs in sorted(chain_groups.items()):
        linker_blocks = []
        current = []
        for ridx in idxs:
            if all_residues[ridx]["pfam_domain"] == "Linker":
                current.append(ridx)
            elif current:
                linker_blocks.append(current)
                current = []
        if current:
            linker_blocks.append(current)

        for bidx, block in enumerate(linker_blocks):
            units.append({
                "unit_id": f"chain:{chain}|linker:{bidx}",
                "unit_type": "linker",
                "chain_id": chain,
                "pfam_id": None,
                "residue_indices": block,
            })

        if include_chain_units:
            units.append({
                "unit_id": f"chain:{chain}|whole_chain",
                "unit_type": "chain",
                "chain_id": chain,
                "pfam_id": None,
                "residue_indices": idxs,
            })

    pfam_ids = sorted({u["pfam_id"] for u in units if u["unit_type"] == "pfam_domain"})
    pfam_to_mask = {p: i + 2 for i, p in enumerate(pfam_ids)}

    for u in units:
        for ridx in u["residue_indices"]:
            if residue_unit_id[ridx] is None:
                residue_unit_id[ridx] = u["unit_id"]
                if u["unit_type"] == "pfam_domain":
                    residue_domain_id[ridx] = u["pfam_id"]
                    domain_mask[ridx] = pfam_to_mask[u["pfam_id"]]
                elif u["unit_type"] == "linker":
                    residue_domain_id[ridx] = "Linker"
                    domain_mask[ridx] = 1
                else:
                    residue_domain_id[ridx] = "Chain"
                    domain_mask[ridx] = 1

    for i in range(len(all_residues)):
        if residue_unit_id[i] is None:
            residue_unit_id[i] = f"chain:{all_residues[i]['chain']}|fallback"
            residue_domain_id[i] = "Linker"
            domain_mask[i] = 1

    return units, residue_domain_id, residue_unit_id, domain_mask


def select_residue_subset(all_residues, dists_to_ligand, contact_residue_mask, pocket_residue_mask, max_seq_len):
    n = len(all_residues)
    if n <= max_seq_len:
        return list(range(n))

    contact_idxs = set(np.where(contact_residue_mask)[0].tolist())
    pocket_idxs = set(np.where(pocket_residue_mask)[0].tolist())

    selected = set(contact_idxs)
    if len(selected) < max_seq_len:
        for idx in sorted(pocket_idxs, key=lambda x: dists_to_ligand[x]):
            if len(selected) >= max_seq_len:
                break
            selected.add(idx)

    if len(selected) < max_seq_len:
        for idx in np.argsort(dists_to_ligand).tolist():
            if len(selected) >= max_seq_len:
                break
            selected.add(int(idx))

    if len(selected) > max_seq_len:
        forced = sorted(contact_idxs)
        remaining = [i for i in sorted(selected, key=lambda x: dists_to_ligand[x]) if i not in contact_idxs]
        selected = set((forced + remaining)[:max_seq_len])

    return sorted(selected)


def reindex_after_subset(all_residues, selected_indices, *arrays):
    subset_residues = [all_residues[i] for i in selected_indices]
    out_arrays = []
    for arr in arrays:
        out_arrays.append(arr[selected_indices])
    old_to_new = {old: new for new, old in enumerate(selected_indices)}
    return subset_residues, out_arrays, old_to_new


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return None
    return v / n


def bootstrap_frame_stability(coords, pocket_center, n_iters):
    # Stub: deterministic score derived from coordinate covariance condition.
    if coords.shape[0] < 3:
        return 0.0
    cov = np.cov(coords.T)
    vals, _ = np.linalg.eigh(cov)
    vals = np.sort(vals)[::-1]
    if vals[0] <= 1e-8:
        return 0.0
    ratio = float(vals[1] / (vals[0] + 1e-8))
    return float(max(0.0, min(1.0, ratio)))


def build_unit_frame(unit, residues, pocket_center, args):
    idxs = unit["residue_indices"]
    coords = np.array([residues[i]["coord"] for i in idxs], dtype=np.float32)
    if coords.shape[0] == 0:
        return {"R": np.eye(3, dtype=np.float32), "t": np.zeros(3, dtype=np.float32), "stable": False,
                "stability_score": 0.0, "num_residues": 0, "source": "failed"}

    local_mask = np.linalg.norm(coords - pocket_center[None, :], axis=1) <= float(args.frame_radius)
    local_coords = coords[local_mask]
    source = "local"
    if local_coords.shape[0] < args.min_frame_residues:
        local_coords = coords
        source = "whole_unit"

    if local_coords.shape[0] < args.min_frame_residues:
        return {"R": np.eye(3, dtype=np.float32), "t": coords.mean(axis=0), "stable": False,
                "stability_score": 0.0, "num_residues": int(local_coords.shape[0]), "source": "failed"}

    centroid = local_coords.mean(axis=0)
    centered = local_coords - centroid
    cov = centered.T @ centered / max(1, centered.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    x_axis = _normalize(vecs[:, 0])
    y_tmp = vecs[:, 1] - np.dot(vecs[:, 1], x_axis) * x_axis
    y_axis = _normalize(y_tmp)
    if x_axis is None or y_axis is None:
        return {"R": np.eye(3, dtype=np.float32), "t": centroid.astype(np.float32), "stable": False,
                "stability_score": 0.0, "num_residues": int(local_coords.shape[0]), "source": "failed"}

    z_axis = _normalize(np.cross(x_axis, y_axis))
    if z_axis is None:
        return {"R": np.eye(3, dtype=np.float32), "t": centroid.astype(np.float32), "stable": False,
                "stability_score": 0.0, "num_residues": int(local_coords.shape[0]), "source": "failed"}
    y_axis = _normalize(np.cross(z_axis, x_axis))
    if y_axis is None:
        return {"R": np.eye(3, dtype=np.float32), "t": centroid.astype(np.float32), "stable": False,
                "stability_score": 0.0, "num_residues": int(local_coords.shape[0]), "source": "failed"}

    to_pocket = pocket_center - centroid
    if np.dot(x_axis, to_pocket) < 0:
        x_axis = -x_axis
        y_axis = -y_axis

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0

    det_ok = abs(np.linalg.det(R) - 1.0) < 5e-2
    ortho_ok = np.allclose(R.T @ R, np.eye(3), atol=5e-2)
    eig_ratio = float(vals[1] / (vals[0] + 1e-8)) if vals[0] > 1e-8 else 0.0
    no_nan = np.isfinite(R).all() and np.isfinite(centroid).all()
    boot = bootstrap_frame_stability(local_coords, pocket_center, args.frame_bootstrap_iters)
    stable = bool(local_coords.shape[0] >= args.min_frame_residues and eig_ratio >= args.frame_eigen_ratio_min and det_ok and ortho_ok and no_nan)
    score = float(min(1.0, max(0.0, 0.5 * eig_ratio + 0.5 * boot)))

    return {
        "R": R.astype(np.float32),
        "t": centroid.astype(np.float32),
        "stable": stable,
        "stability_score": score,
        "num_residues": int(local_coords.shape[0]),
        "source": source if stable else "failed",
    }


def make_T(R, t):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float32)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def compose_T(A, B):
    return A @ B


def relative_T(F_i, F_j):
    return invert_T(F_i) @ F_j


def validate_T(T):
    if T.shape != (4, 4):
        return False
    if not np.isfinite(T).all():
        return False
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-2):
        return False
    if abs(np.linalg.det(R) - 1.0) > 1e-2:
        return False
    return True


def enumerate_candidate_pairs(units, residue_coords, pocket_center, args):
    unit_info = []
    for i, u in enumerate(units):
        idxs = u["residue_indices"]
        coords = residue_coords[idxs]
        if coords.shape[0] == 0:
            min_d = 1e9
            n_pocket = 0
        else:
            d = np.linalg.norm(coords - pocket_center[None, :], axis=1)
            min_d = float(d.min())
            n_pocket = int((d <= args.pocket_radius).sum())
        score = -min_d + 0.1 * n_pocket
        unit_info.append({"idx": i, "min_d": min_d, "num_pocket": n_pocket, "score": score})

    cands = [u for u in unit_info if u["num_pocket"] > 0]
    if not cands:
        cands = sorted(unit_info, key=lambda x: x["min_d"])[:max(1, args.max_domain_pairs)]

    cand_indices = [c["idx"] for c in sorted(cands, key=lambda x: (-x["score"], x["idx"]))]

    pair_records = []
    for i, j in combinations(cand_indices, 2):
        ui, uj = units[i], units[j]
        if ui["unit_type"] == "chain" and uj["unit_type"] == "chain":
            ptype = "chain_pair"
        elif ui["unit_type"] == "linker" or uj["unit_type"] == "linker":
            ptype = "linker_pair"
        elif ui["chain_id"] == uj["chain_id"]:
            ptype = "intra_chain_domain_pair"
        else:
            ptype = "mixed"
        prior = 0.5 * (unit_info[i]["score"] + unit_info[j]["score"])
        pair_records.append((i, j, ptype, prior))

    if args.allow_single_domain_candidates or len(cand_indices) == 1:
        for i in cand_indices:
            pair_records.append((i, -1, "single_unit", unit_info[i]["score"]))

    pair_records = sorted(pair_records, key=lambda x: (-x[3], x[0], x[1]))[:max(1, args.max_domain_pairs)]
    return pair_records, unit_info


def classify_binding_case_v2(units, contact_residue_mask):
    contact_units = []
    for i, u in enumerate(units):
        idxs = u["residue_indices"]
        if len(idxs) == 0:
            continue
        if contact_residue_mask[np.array(idxs)].any():
            contact_units.append((i, u))

    if not contact_units:
        return "no_contact"

    types = [u["unit_type"] for _, u in contact_units]
    chains = {u["chain_id"] for _, u in contact_units}
    pfam_units = [u for _, u in contact_units if u["unit_type"] == "pfam_domain"]
    linker_units = [u for _, u in contact_units if u["unit_type"] == "linker"]

    if len(pfam_units) == 1 and len(contact_units) == 1:
        return "single_domain"
    if len(pfam_units) == 1 and len(linker_units) >= 1 and len(chains) == 1:
        return "single_domain_with_linker_contact"
    if len(pfam_units) >= 2 and len(chains) == 1:
        return "intra_chain_domain_interface"
    if len(pfam_units) == 0 and len(linker_units) > 0:
        return "linker_binder"
    if len(chains) >= 2 and len(pfam_units) >= 2:
        by_chain = {}
        for u in pfam_units:
            by_chain.setdefault(u["chain_id"], 0)
            by_chain[u["chain_id"]] += 1
        has_cross_chain = len(by_chain) >= 2
        has_intra_chain_multi_domain = any(v >= 2 for v in by_chain.values())
        if has_cross_chain and has_intra_chain_multi_domain:
            return "mixed_domain_chain_interface"
        if has_cross_chain:
            return "chain_interface"
    return "unknown"


def build_sample_dict(row, prot_file, lig_file, residues, sequence, ca_coords, domain_mask, residue_domain_id,
                               residue_unit_id, lig_feat, known_pocket_center, pocket_residue_mask, contact_residue_mask,
                               units, frame_data, pair_data, binding_case_v2, dists_to_ligand):
    unit_ids = [u["unit_id"] for u in units]
    unit_types = [u["unit_type"] for u in units]
    unit_chain_ids = [u["chain_id"] for u in units]
    unit_pfam_ids = [u["pfam_id"] for u in units]
    unit_residue_indices = [u["residue_indices"] for u in units]

    unit_contact_mask = torch.tensor([
        bool(contact_residue_mask[np.array(u["residue_indices"])].any()) if len(u["residue_indices"]) else False
        for u in units
    ], dtype=torch.bool)
    unit_min_ligand_dist = torch.tensor([
        float(dists_to_ligand[np.array(u["residue_indices"])].min()) if len(u["residue_indices"]) else 1e9
        for u in units
    ], dtype=torch.float32)

    frame_R = torch.tensor(np.stack([f["R"] for f in frame_data], axis=0), dtype=torch.float32)
    frame_t = torch.tensor(np.stack([f["t"] for f in frame_data], axis=0), dtype=torch.float32)
    frame_stable = torch.tensor([f["stable"] for f in frame_data], dtype=torch.bool)
    frame_stability_score = torch.tensor([f["stability_score"] for f in frame_data], dtype=torch.float32)
    frame_num_residues = torch.tensor([f["num_residues"] for f in frame_data], dtype=torch.long)
    frame_source = [f["source"] for f in frame_data]

    pair_i = torch.tensor([p[0] for p in pair_data], dtype=torch.long)
    pair_j = torch.tensor([p[1] for p in pair_data], dtype=torch.long)
    pair_type = [p[2] for p in pair_data]
    pair_prior = torch.tensor([p[3] for p in pair_data], dtype=torch.float32)
    pair_true = torch.tensor([p[4] for p in pair_data], dtype=torch.bool)
    T_ij = torch.tensor(np.stack([p[5] for p in pair_data], axis=0), dtype=torch.float32)
    T_ji = torch.tensor(np.stack([p[6] for p in pair_data], axis=0), dtype=torch.float32)
    pair_frame_stable = torch.tensor([p[7] for p in pair_data], dtype=torch.bool)

    true_contact_unit_indices = torch.where(unit_contact_mask)[0].long()
    true_contact_pair_indices = torch.where(pair_true)[0].long()
    contact_residue_indices = torch.tensor(np.where(contact_residue_mask)[0], dtype=torch.long)

    sample_dict = {
        "metadata": {
            "Assembly_ID": str(row["Assembly_ID"]),
            "Ligand_file": str(row["Ligand_file"]),
            "Ligand_ID": str(row["Ligand_ID"]),
            "smiles": str(row.get("smiles", "")),
            "prot_file": str(prot_file),
            "lig_file": str(lig_file),
            "source": "Q-BioLiP",
            "is_holo_supervision": True,
        },
        "protein": {
            "sequence": sequence,
            "ca_coord": torch.tensor(ca_coords, dtype=torch.float32),
            "residue_chain": [r["chain"] for r in residues],
            "residue_number": torch.tensor([r["resid"] for r in residues], dtype=torch.long),
            "residue_name": [r["resname"] for r in residues],
            "residue_domain_id": list(residue_domain_id),
            "residue_unit_id": list(residue_unit_id),
            "domain_mask": torch.tensor(domain_mask, dtype=torch.long),
        },
        "ligand": {
            "atom_coord": torch.tensor(lig_feat["atom_coord"], dtype=torch.float32),
            "atom_symbol": lig_feat["atom_symbol"],
            "ligand_center": torch.tensor(lig_feat["ligand_center"], dtype=torch.float32),
            "smiles": str(row.get("smiles", "")),
        },
        "pocket": {
            "known_pocket_center": torch.tensor(known_pocket_center, dtype=torch.float32),
            "pocket_radius": float(args.pocket_radius),
            "pocket_residue_mask": torch.tensor(pocket_residue_mask, dtype=torch.bool),
            "contact_residue_mask": torch.tensor(contact_residue_mask, dtype=torch.bool),
        },
        "units": {
            "unit_ids": unit_ids,
            "unit_types": unit_types,
            "unit_chain_ids": unit_chain_ids,
            "unit_pfam_ids": unit_pfam_ids,
            "unit_residue_indices": unit_residue_indices,
            "unit_contact_mask": unit_contact_mask,
            "unit_min_ligand_dist": unit_min_ligand_dist,
        },
        "frames": {
            "frame_R": frame_R,
            "frame_t": frame_t,
            "frame_stable": frame_stable,
            "frame_stability_score": frame_stability_score,
            "frame_num_residues": frame_num_residues,
            "frame_source": frame_source,
        },
        "candidate_pairs": {
            "pair_unit_i": pair_i,
            "pair_unit_j": pair_j,
            "pair_type": pair_type,
            "pair_prior_score": pair_prior,
            "pair_is_true_contact": pair_true,
            "T_ij": T_ij,
            "T_ji": T_ji,
            "pair_frame_stable": pair_frame_stable,
        },
        "supervision": {
            "binding_case_v2": binding_case_v2,
            "true_contact_unit_indices": true_contact_unit_indices,
            "true_contact_pair_indices": true_contact_pair_indices,
            "contact_residue_indices": contact_residue_indices,
            "ligand_contact_dist": torch.tensor(dists_to_ligand, dtype=torch.float32),
        },
    }
    return sample_dict


def safe_torch_save(obj, path):
    try:
        torch.save(obj, path)
        return True
    except Exception:
        return False


def process_single_entry(row, args, pfam_subset):
    try:
        pdbid = row["Assembly_ID"]
        lig_name = row["Ligand_file"]
        safe_lig = safe_id(lig_name)

        if pfam_subset is None or len(pfam_subset) == 0:
            return {"ok": False, "Assembly_ID": pdbid, "Ligand_file": lig_name, "filter_reason": "missing_pfam"}

        prot_file = os.path.join(args.biolip_pdb_root, f"{pdbid}.pdb")
        lig_file = os.path.join(args.biolip_lig_root, f"{lig_name}.pdb")
        if not os.path.exists(prot_file):
            return {"ok": False, "Assembly_ID": pdbid, "Ligand_file": lig_name, "filter_reason": "missing_protein_file"}
        if not os.path.exists(lig_file):
            return {"ok": False, "Assembly_ID": pdbid, "Ligand_file": lig_name, "filter_reason": "missing_ligand_file"}

        lig_feat = compute_ligand_features(lig_file)
        if lig_feat is None:
            return {"ok": False, "Assembly_ID": pdbid, "Ligand_file": lig_name, "filter_reason": "invalid_ligand"}

        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure(pdbid, prot_file)
            model = next(iter(structure))
        except Exception:
            return {"ok": False, "Assembly_ID": pdbid, "Ligand_file": lig_name, "filter_reason": "invalid_protein"}

        domain_lookup = build_domain_lookup(pfam_subset)
        all_residues = extract_standard_residues(model, domain_lookup, COMMON_AAS)
        if not all_residues:
            return {"ok": False, "Assembly_ID": pdbid, "Ligand_file": lig_name, "filter_reason": "no_standard_residues"}

        prot_coords = np.array([r["coord"] for r in all_residues], dtype=np.float32)
        dists_to_ligand, contact_mask, pocket_center, pocket_mask = compute_contact_and_pocket_masks(
            prot_coords, lig_feat["atom_coord"], args
        )

        selected_indices = select_residue_subset(
            all_residues, dists_to_ligand, contact_mask, pocket_mask, args.max_seq_len
        )
        if len(selected_indices) < 10:
            return {"ok": False, "Assembly_ID": pdbid, "Ligand_file": lig_name, "filter_reason": "too_few_selected_residues"}

        subset_residues, subset_arrays, old_to_new = reindex_after_subset(
            all_residues, selected_indices, dists_to_ligand, contact_mask, pocket_mask
        )
        subset_dists, subset_contact_mask, subset_pocket_mask = subset_arrays

        subset_pdb_name = os.path.join(args.data_root, 'prot_subset_pdb', f"{safe_lig}_subset.pdb")
        save_subset_pdb(model, selected_indices, all_residues, subset_pdb_name)
        units, residue_domain_id, residue_unit_id, domain_mask = build_units(subset_residues, include_chain_units=True)
        if len(units) == 0:
            return {"ok": False, "Assembly_ID": pdbid, "Ligand_file": lig_name, "filter_reason": "no_units"}

        # Remove tiny non-contact units after truncation.
        filtered_units = []
        for u in units:
            idxs = u["residue_indices"]
            is_contact = bool(subset_contact_mask[np.array(idxs)].any()) if idxs else False
            if len(idxs) < args.min_frame_residues and not is_contact and u["unit_type"] != "chain":
                continue
            filtered_units.append(u)
        units = filtered_units
        if len(units) == 0:
            return {"ok": False, "Assembly_ID": pdbid, "Ligand_file": lig_name, "filter_reason": "no_units"}

        residue_coords = np.array([r["coord"] for r in subset_residues], dtype=np.float32)
        frame_data = [build_unit_frame(u, subset_residues, pocket_center, args) for u in units]
        stable_count = int(sum(1 for f in frame_data if f["stable"]))

        pair_records, unit_info = enumerate_candidate_pairs(units, residue_coords, pocket_center, args)
        if len(pair_records) == 0:
            return {"ok": False, "Assembly_ID": pdbid, "Ligand_file": lig_name, "filter_reason": "no_candidate_pairs"}

        unit_contact = [
            bool(subset_contact_mask[np.array(u["residue_indices"])].any()) if len(u["residue_indices"]) else False
            for u in units
        ]
        pair_data = []
        for i, j, ptype, prior in pair_records:
            Fi = make_T(frame_data[i]["R"], frame_data[i]["t"])
            if j == -1:
                T_ij = np.eye(4, dtype=np.float32)
                T_ji = np.eye(4, dtype=np.float32)
                true_pair = bool(unit_contact[i] and sum(unit_contact) == 1)
                frame_ok = frame_data[i]["stable"]
            else:
                Fj = make_T(frame_data[j]["R"], frame_data[j]["t"])
                T_ij = relative_T(Fi, Fj)
                T_ji = relative_T(Fj, Fi)
                true_pair = bool(unit_contact[i] and unit_contact[j])
                frame_ok = bool(frame_data[i]["stable"] and frame_data[j]["stable"])
            pair_data.append((i, j, ptype, float(prior), true_pair, T_ij, T_ji, frame_ok))

        if stable_count == 0:
            return {"ok": False, "Assembly_ID": pdbid, "Ligand_file": lig_name, "filter_reason": "no_valid_frames"}

        seq = "".join(THREE_TO_ONE.get(r["resname"], "X") for r in subset_residues)
        sample_dict = build_sample_dict(
            row=row,
            prot_file=subset_pdb_name,
            lig_file=lig_file,
            residues=subset_residues,
            sequence=seq,
            ca_coords=residue_coords,
            domain_mask=domain_mask,
            residue_domain_id=residue_domain_id,
            residue_unit_id=residue_unit_id,
            lig_feat=lig_feat,
            known_pocket_center=pocket_center,
            pocket_residue_mask=subset_pocket_mask,
            contact_residue_mask=subset_contact_mask,
            units=units,
            frame_data=frame_data,
            pair_data=pair_data,
            binding_case_v2=classify_binding_case_v2(units, subset_contact_mask),
            dists_to_ligand=subset_dists,
        )

        motordock_file = os.path.join(args.data_root, 'sample_dict', f"{safe_lig}.pt")

        if not safe_torch_save(sample_dict, motordock_file):
            return {"ok": False, "Assembly_ID": pdbid, "Ligand_file": lig_name, "filter_reason": "save_failed"}

        if not safe_torch_save(torch.tensor(domain_mask, dtype=torch.long), os.path.join(args.data_root, 'domain_masks', f"{safe_lig}_domain_mask.pt")):
            return {"ok": False, "Assembly_ID": pdbid, "Ligand_file": lig_name, "filter_reason": "save_failed"}

        if args.save_json_debug:
            debug_file = os.path.join(args.data_root, 'frames', f"{safe_lig}_frames.json")
            with open(debug_file, "w", encoding="utf-8") as f:
                json.dump({
                    "frames": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in fr.items()} for fr in frame_data],
                    "pairs": [{"i": p[0], "j": p[1], "type": p[2], "score": p[3]} for p in pair_data],
                }, f)

        num_contact_units = int(sample_dict["units"]["unit_contact_mask"].sum().item())
        true_contact_unit_ids = [sample_dict["units"]["unit_ids"][i] for i in sample_dict["supervision"]["true_contact_unit_indices"].tolist()]
        true_contact_pair_ids = [
            f"{pair_data[i][0]}_{pair_data[i][1]}" for i in sample_dict["supervision"]["true_contact_pair_indices"].tolist()
        ]
        binding_case_v2 = sample_dict["supervision"]["binding_case_v2"]
        has_chain_interface_contact = "chain_interface" in binding_case_v2 or "mixed" in binding_case_v2
        has_interface_contact = num_contact_units >= 2
        has_linker_contact = any(
            utype == "linker" and bool(mask)
            for utype, mask in zip(sample_dict["units"]["unit_types"], sample_dict["units"]["unit_contact_mask"].tolist())
        )

        csv_row = {
            "Assembly_ID": row["Assembly_ID"],
            "Ligand_file": row["Ligand_file"],
            "Ligand_ID": row["Ligand_ID"],
            "Binding_residue": row.get("Binding_residue", ""),
            "smiles": row.get("smiles", ""),
            "type": row.get("type", "Unknown"),
            "contacting_domains": row.get("contacting_domains", np.nan),
            "total_domains": row.get("total_domains", np.nan),
            "prot_file": subset_pdb_name,
            "motordock_file": motordock_file,
            "num_units": len(units),
            "num_candidate_pairs": len(pair_data),
            "num_stable_frames": stable_count,
            "num_contact_units": num_contact_units,
            "true_contact_unit_ids": "|".join(true_contact_unit_ids),
            "true_contact_pair_ids": "|".join(true_contact_pair_ids),
            "binding_case_v2": binding_case_v2,
            "has_interface_contact": bool(has_interface_contact),
            "has_chain_interface_contact": bool(has_chain_interface_contact),
            "has_linker_contact": bool(has_linker_contact),
            "has_valid_ligand": True,
            "has_valid_frames": stable_count > 0,
            "selected_residue_count": len(subset_residues),
            "original_residue_count": len(all_residues),
            "ligand_atom_count": lig_feat["ligand_atom_count"],
            "pocket_center_x": float(pocket_center[0]),
            "pocket_center_y": float(pocket_center[1]),
            "pocket_center_z": float(pocket_center[2]),
            "filter_reason": "",
        }
        return {"ok": True, "row": csv_row}
    except Exception:
        return {"ok": False, "Assembly_ID": row.get("Assembly_ID", ""), "Ligand_file": row.get("Ligand_file", ""), "filter_reason": "unknown_error"}

def gen_pretrain(args):
    os.makedirs(args.data_root, exist_ok=True)
    os.makedirs(os.path.join(args.data_root, 'frames'), exist_ok=True)
    os.makedirs(os.path.join(args.data_root, 'candidates'), exist_ok=True)
    os.makedirs(os.path.join(args.data_root, 'domain_masks'), exist_ok=True)
    os.makedirs(os.path.join(args.data_root, 'prot_subset_pdb'), exist_ok=True)
    os.makedirs(os.path.join(args.data_root, 'sample_dict'), exist_ok=True)

    pfam_lst = pd.read_csv(args.pfam_lst_path, sep="\t")
    pfam_lst = pfam_lst.dropna(subset=["AUTH_PDBRES_START", "AUTH_PDBRES_END"])
    pfam_groups = {k: v for k, v in pfam_lst.groupby("PDB")}

    biolip_df = pd.read_csv(args.biolip_df_path)
    if args.end_idx != -1:
        biolip_df = biolip_df.iloc[args.start_idx:args.end_idx]

    indexed_rows = list(biolip_df.iterrows())
    tasks = []
    task_indices = []
    for idx, row in indexed_rows:
        pdb_key = str(row["Assembly_ID"]).split("_")[0]
        tasks.append(delayed(process_single_entry)(row, args, pfam_groups.get(pdb_key)))
        task_indices.append(idx)

    num_cores = resolve_num_workers(args.num_workers)
    results = Parallel(n_jobs=num_cores)(tqdm(tasks, total=len(tasks)))

    success_items = []
    fail_items = []
    for orig_idx, res in zip(task_indices, results):
        if res.get("ok", False):
            success_items.append((orig_idx, res["row"]))
        else:
            fail_items.append({
                "Assembly_ID": res.get("Assembly_ID", ""),
                "Ligand_file": res.get("Ligand_file", ""),
                "filter_reason": res.get("filter_reason", "unknown_error"),
            })

    success_items.sort(key=lambda x: x[0])
    success_rows = [x[1] for x in success_items]

    suffix = start_end_suffix(args)
    success_csv = os.path.join(args.data_root, args.motordock_csv_name.replace(".csv", f"{suffix}.csv"))
    fail_csv = os.path.join(args.data_root, f"01_qbiolip_failed{suffix}.csv")

    pd.DataFrame(success_rows).to_csv(success_csv, index=False)
    pd.DataFrame(fail_items, columns=["Assembly_ID", "Ligand_file", "filter_reason"]).to_csv(fail_csv, index=False)

    print(f"MotorDock pretrain done. success={len(success_rows)} fail={len(fail_items)}")
    print(f"Saved: {success_csv}")
    print(f"Saved: {fail_csv}")


def _tensor_has_invalid(x):
    if isinstance(x, torch.Tensor):
        return (not torch.isfinite(x).all().item())
    return False


def _run_internal_asserts():
    R = np.eye(3, dtype=np.float32)
    t = np.array([1, 2, 3], dtype=np.float32)
    T = make_T(R, t)
    T_inv = invert_T(T)
    I = compose_T(T, T_inv)
    assert np.allclose(I, np.eye(4), atol=1e-5), "T invert consistency failed"
    T_rel = relative_T(T, T)
    assert np.allclose(T_rel, np.eye(4), atol=1e-5), "relative_T self consistency failed"


def validate_pretrain(args):
    _run_internal_asserts()

    suffix = start_end_suffix(args)
    motordock_csv = os.path.join(args.data_root, args.motordock_csv_name.replace(".csv", f"{suffix}.csv"))
    if not os.path.exists(motordock_csv):
        print(f"Missing motordock CSV: {motordock_csv}")
        return

    df = pd.read_csv(motordock_csv)
    reports = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = row.get("motordock_file", "")
        out = {
            "motordock_file": path,
            "is_valid": False,
            "error_reason": "",
            "num_residues": 0,
            "num_units": 0,
            "num_pairs": 0,
            "num_stable_frames": 0,
        }
        try:
            if not path or not os.path.exists(path):
                out["error_reason"] = "missing_file"
                reports.append(out)
                continue

            x = torch.load(path, map_location="cpu")
            req_top = ["metadata", "protein", "ligand", "pocket", "units", "frames", "candidate_pairs", "supervision"]
            for k in req_top:
                if k not in x:
                    raise ValueError(f"missing_key:{k}")

            ca = x["protein"]["ca_coord"]
            lig = x["ligand"]["atom_coord"]
            dm = x["protein"]["domain_mask"]
            frame_R = x["frames"]["frame_R"]
            frame_t = x["frames"]["frame_t"]
            T_ij = x["candidate_pairs"]["T_ij"]
            true_pair_idx = x["supervision"]["true_contact_pair_indices"]

            N = int(ca.shape[0])
            U = int(frame_R.shape[0])
            P = int(T_ij.shape[0])

            if ca.ndim != 2 or ca.shape[1] != 3:
                raise ValueError("bad_shape:protein.ca_coord")
            if lig.ndim != 2 or lig.shape[1] != 3:
                raise ValueError("bad_shape:ligand.atom_coord")
            if dm.ndim != 1 or dm.shape[0] != N:
                raise ValueError("bad_shape:protein.domain_mask")
            if frame_R.shape != (U, 3, 3):
                raise ValueError("bad_shape:frames.frame_R")
            if frame_t.shape != (U, 3):
                raise ValueError("bad_shape:frames.frame_t")
            if T_ij.shape != (P, 4, 4):
                raise ValueError("bad_shape:candidate_pairs.T_ij")
            if P < 1:
                raise ValueError("empty_candidate_pairs")

            if _tensor_has_invalid(ca) or _tensor_has_invalid(lig) or _tensor_has_invalid(frame_R) or _tensor_has_invalid(frame_t) or _tensor_has_invalid(T_ij):
                raise ValueError("nan_or_inf")

            for p in range(P):
                if not validate_T(T_ij[p].numpy()):
                    raise ValueError("invalid_transform")

            frame_stable = x["frames"]["frame_stable"]
            stable_count = int(frame_stable.sum().item())
            if bool(row.get("has_valid_frames", True)) and stable_count < 1:
                raise ValueError("has_valid_frames_but_no_stable")

            if true_pair_idx.numel() > 0:
                if int(true_pair_idx.max().item()) >= P:
                    raise ValueError("true_contact_pair_indices_oob")

            out["is_valid"] = True
            out["num_residues"] = N
            out["num_units"] = int(len(x["units"]["unit_ids"]))
            out["num_pairs"] = P
            out["num_stable_frames"] = stable_count
        except Exception as e:
            out["error_reason"] = str(e)
        reports.append(out)

    report_path = os.path.join(args.data_root, f"validation_report{suffix}.csv")
    pd.DataFrame(reports).to_csv(report_path, index=False)
    valid_n = sum(1 for r in reports if r["is_valid"])
    print(f"Validation complete: valid={valid_n}/{len(reports)}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    if args.task == "gen_pretrain":
        gen_pretrain(args)
    elif args.task == "validate_pretrain":
        validate_pretrain(args)
