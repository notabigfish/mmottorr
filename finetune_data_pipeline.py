import os
import json
import argparse
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from rdkit import Chem, RDLogger
from Bio.PDB import PDBParser
from scipy.spatial import distance
from joblib import Parallel, delayed

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--start_idx', type=int, default=0, help='Start index for processing')
parser.add_argument('--end_idx', type=int, default=-1, help='End index for processing')
parser.add_argument('--pfam_lst_path', type=str, default='/rds/homes/s/sxz325/shuo/7788/multi-domain-enzyme/data/pdb_pfam_mapping_0517.tsv', help='Path to PFAM list file')
parser.add_argument('--max_seq_len', type=int, default=1022, help='Maximum sequence length for processing')
parser.add_argument('--pdbbind-root', default='/rds/homes/s/sxz325/shuo/7788/multi-domain-enzyme/data/pdbbind2020')
parser.add_argument('--index-file', default=None)
parser.add_argument('--output-dir', default='./data/pdbbind')
parser.add_argument('--train-split', default='/rds/homes/s/sxz325/shuo/7788/multi-domain-enzyme/data/timesplit_no_lig_overlap_train')
parser.add_argument('--val-split', default='/rds/homes/s/sxz325/shuo/7788/multi-domain-enzyme/data/timesplit_no_lig_overlap_val')
parser.add_argument('--test-split', default='/rds/homes/s/sxz325/shuo/7788/multi-domain-enzyme/data/timesplit_test')

parser.add_argument('--require-pocket', action='store_true', default=True,
                    help='Require {pdb_id}_pocket.pdb to exist.')
parser.add_argument('--crop-source', type=str, default='pocket',
                    choices=['pocket', 'ligand'],
                    help='Use pocket-centered or ligand-centered residue cropping.')
parser.add_argument('--pocket-radius', type=float, default=15.0,
                    help='Radius in Angstrom for selecting domain/chain units near a pocket.')
parser.add_argument('--contact-threshold', type=float, default=6.0,
                    help='Distance threshold in Angstrom for legacy ligand-contact classification.')
parser.add_argument('--frame-min-residues', type=int, default=8,
                    help='Minimum residues required to build a local frame.')
parser.add_argument('--frame-min-ca-atoms', type=int, default=8,
                    help='Minimum CA atoms required to build a local frame.')
parser.add_argument('--frame-eig-ratio-threshold', type=float, default=0.15,
                    help='Minimum PCA eigenvalue ratio lambda2/lambda1 for stable frame construction.')

args = parser.parse_args()
if args.index_file is None:
    args.index_file = os.path.join(args.pdbbind_root, 'index', 'INDEX_general_PL_data.2020')

COMMON_AAS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

def load_split_ids(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return handle.read().splitlines()

def assign_split(row, train_ids, val_ids, test_ids):
    if row['Assembly_ID'] in train_ids:
        return 'train'
    if row['Assembly_ID'] in val_ids:
        return 'val'
    if row['Assembly_ID'] in test_ids:
        return 'test'
    return 'unknown'

def format_pfam_domains(group):
    rows = [[
        row['PFAM_ACCESSION'],
        int(row['AUTH_PDBRES_START']),
        int(row['AUTH_PDBRES_END']),
        row['CHAIN']
    ] for _, row in group.iterrows()]
    return pd.DataFrame(rows, columns=['pfam_id', 'start', 'end', 'chain_id'])

def load_index_file(index_file):
    data = []
    with open(index_file, 'r', encoding='utf-8') as handle:
        for line in handle:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 5:
                data.append({'pdb_id': parts[0], 'affinity': float(parts[3])})
    return pd.DataFrame(data)

def get_residue_key(chain_id, residue):
    hetflag, resseq, icode = residue.get_id()
    _ = hetflag
    return (chain_id, int(resseq), icode.strip())

def get_subset_info(indices, all_residues):
    ca_coord = []
    seq_list = []
    for idx in indices:
        residue = all_residues[idx]['obj']
        res_name_3 = residue.get_resname()
        res_name_1 = THREE_TO_ONE.get(res_name_3, 'X')
        if res_name_1 == 'X':
            print(f'Warning: Unknown residue {res_name_3} at index {idx}. Using X.')
        if 'CA' in residue:
            ca_coord.append(residue['CA'].get_coord())
            seq_list.append(res_name_1)
        else:
            print(f'Warning: Residue missing CA atom at index {idx}.')
    if not ca_coord:
        return torch.zeros((0, 3), dtype=torch.float32), ''
    return torch.tensor(np.array(ca_coord), dtype=torch.float32), ''.join(seq_list)

def parse_pocket_pdb(pocket_file):
    '''
    Parse {pdb_id}_pocket.pdb.
    Return:
        pocket_reskeys: set of (chain_id, resid, icode)
        pocket_ca_coords: np.ndarray [n_pocket_res, 3]
        pocket_atom_coords: np.ndarray [n_pocket_atoms, 3]
        pocket_center_ca: np.ndarray [3]
        pocket_center_atom: np.ndarray [3]
    '''
    parser_local = PDBParser(QUIET=True)
    structure = parser_local.get_structure('pocket', pocket_file)
    model = next(iter(structure))

    pocket_reskeys = set()
    pocket_ca_coords = []
    pocket_atom_coords = []
    for chain in model:
        chain_id = chain.get_id()
        for residue in chain:
            if residue.get_resname() not in COMMON_AAS:
                continue
            pocket_reskeys.add(get_residue_key(chain_id, residue))
            if 'CA' in residue:
                pocket_ca_coords.append(residue['CA'].get_coord())
            for atom in residue:
                pocket_atom_coords.append(atom.get_coord())

    pocket_ca_coords = np.array(pocket_ca_coords) if pocket_ca_coords else np.zeros((0, 3), dtype=np.float32)
    pocket_atom_coords = np.array(pocket_atom_coords) if pocket_atom_coords else np.zeros((0, 3), dtype=np.float32)

    pocket_center_ca = None
    pocket_center_atom = None
    if pocket_ca_coords.shape[0] > 0:
        pocket_center_ca = pocket_ca_coords.mean(axis=0)
    if pocket_atom_coords.shape[0] > 0:
        pocket_center_atom = pocket_atom_coords.mean(axis=0)
    if pocket_center_ca is None and pocket_center_atom is not None:
        pocket_center_ca = pocket_center_atom.copy()

    return {
        'pocket_reskeys': pocket_reskeys,
        'pocket_ca_coords': pocket_ca_coords,
        'pocket_atom_coords': pocket_atom_coords,
        'pocket_center_ca': pocket_center_ca,
        'pocket_center_atom': pocket_center_atom,
    }

def match_pocket_to_selected_residues(subset_residues, pocket_reskeys):
    selected_reskeys = [res['reskey'] for res in subset_residues]
    pocket_selected_indices = [
        i for i, res in enumerate(subset_residues)
        if res['reskey'] in pocket_reskeys
    ]
    pocket_mask_list = [1 if rk in pocket_reskeys else 0 for rk in selected_reskeys]
    pocket_mask_tensor = torch.tensor(pocket_mask_list, dtype=torch.long)
    return selected_reskeys, pocket_selected_indices, pocket_mask_list, pocket_mask_tensor

def build_domain_units(subset_residues):
    units = {}
    for idx, res in enumerate(subset_residues):
        chain_id = res['chain']
        pfam_domain = res['pfam_domain']
        if pfam_domain == 'Linker':
            unit_id = f'{chain_id}_Linker'
            unit_type = 'linker'
        else:
            unit_id = res['cid_domain'] if res['cid_domain'] != 'None' else f'{chain_id}_{pfam_domain}'
            unit_type = 'pfam_domain'

        if unit_id not in units:
            units[unit_id] = {
                'unit_id': unit_id,
                'chain_id': chain_id,
                'pfam_domain': pfam_domain,
                'unit_type': unit_type,
                'residue_indices': [],
                'ca_coords': []
            }
        units[unit_id]['residue_indices'].append(idx)
        units[unit_id]['ca_coords'].append(res['coord'])

    chain_units = {}
    for idx, res in enumerate(subset_residues):
        chain_id = res['chain']
        unit_id = f'CHAIN_{chain_id}'
        if unit_id not in chain_units:
            chain_units[unit_id] = {
                'unit_id': unit_id,
                'chain_id': chain_id,
                'pfam_domain': 'CHAIN',
                'unit_type': 'chain',
                'residue_indices': [],
                'ca_coords': []
            }
        chain_units[unit_id]['residue_indices'].append(idx)
        chain_units[unit_id]['ca_coords'].append(res['coord'])

    out = []
    for u in list(units.values()) + list(chain_units.values()):
        if len(u['ca_coords']) == 0:
            continue
        u['ca_coords'] = np.array(u['ca_coords'])
        out.append(u)
    return out

def enumerate_candidate_pairs(domain_units, pocket_center, radius):
    nearby_units = []
    for unit in domain_units:
        d = np.linalg.norm(unit['ca_coords'] - pocket_center, axis=1)
        if np.any(d <= radius):
            nearby_units.append(unit)

    candidates = []
    if len(nearby_units) == 1:
        u = nearby_units[0]
        candidates.append({
            'pair_id': f"{u['unit_id']}__{u['unit_id']}",
            'unit_a': u['unit_id'],
            'unit_b': u['unit_id'],
            'pair_type': 'single',
            'unit_a_type': u['unit_type'],
            'unit_b_type': u['unit_type'],
            'unit_a_chain': u['chain_id'],
            'unit_b_chain': u['chain_id'],
            'unit_a_num_residues': len(u['residue_indices']),
            'unit_b_num_residues': len(u['residue_indices']),
        })

    for ua, ub in combinations(nearby_units, 2):
        if ua['unit_type'] == 'chain' and ub['unit_type'] == 'chain':
            pair_type = 'chain_pair'
        elif ua['unit_type'] == 'pfam_domain' and ub['unit_type'] == 'pfam_domain':
            pair_type = 'domain_pair'
        else:
            pair_type = 'mixed_pair'
        candidates.append({
            'pair_id': f"{ua['unit_id']}__{ub['unit_id']}",
            'unit_a': ua['unit_id'],
            'unit_b': ub['unit_id'],
            'pair_type': pair_type,
            'unit_a_type': ua['unit_type'],
            'unit_b_type': ub['unit_type'],
            'unit_a_chain': ua['chain_id'],
            'unit_b_chain': ub['chain_id'],
            'unit_a_num_residues': len(ua['residue_indices']),
            'unit_b_num_residues': len(ub['residue_indices']),
        })

    return nearby_units, candidates

def build_local_frame_for_unit(unit, pocket_center, min_residues, eig_ratio_threshold):
    coords = unit['ca_coords']
    if len(coords) < min_residues:
        return None, None, False, 'too_few_residues', np.array([0.0, 0.0, 0.0], dtype=np.float32)

    dists = np.linalg.norm(coords - pocket_center, axis=1)
    weights = 1.0 / (dists + 1e-6)
    weights = weights / (weights.sum() + 1e-12)

    t = np.sum(coords * weights[:, None], axis=0)
    centered = coords - t
    cov = (centered * weights[:, None]).T @ centered

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    x = eigvecs[:, 0]
    y = eigvecs[:, 1]
    y = y - x * np.dot(x, y)
    yn = np.linalg.norm(y)
    if yn < 1e-10:
        return None, t, False, 'degenerate_second_axis', eigvals
    y = y / yn
    z = np.cross(x, y)
    zn = np.linalg.norm(z)
    if zn < 1e-10:
        return None, t, False, 'degenerate_cross_axis', eigvals
    z = z / zn
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-12)

    if np.dot(x, (pocket_center - t)) < 0:
        x = -x
    z = np.cross(x, y)
    z = z / (np.linalg.norm(z) + 1e-12)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-12)

    R = np.stack([x, y, z], axis=1)
    if np.linalg.det(R) < 0:
        z = -z
        y = np.cross(z, x)
        y = y / (np.linalg.norm(y) + 1e-12)
        R = np.stack([x, y, z], axis=1)

    stable = True
    reason = 'ok'
    if eigvals[0] <= 1e-12:
        stable = False
        reason = 'zero_primary_eigenvalue'
    else:
        ratio = eigvals[1] / (eigvals[0] + 1e-12)
        if ratio < eig_ratio_threshold:
            stable = False
            reason = f'eig_ratio_below_threshold:{ratio:.4f}'

    return R, t, stable, reason, eigvals

def make_frame_matrix(R, t):
    F = np.eye(4, dtype=np.float32)
    F[:3, :3] = R.astype(np.float32)
    F[:3, 3] = t.astype(np.float32)
    return F

def invert_transform(F):
    R = F[:3, :3]
    t = F[:3, 3]
    Finv = np.eye(4, dtype=np.float32)
    Finv[:3, :3] = R.T
    Finv[:3, 3] = -R.T @ t
    return Finv

def compute_pair_transform(frame_a, frame_b):
    return invert_transform(frame_a) @ frame_b

def classify_motordock_type(subset_residues, pocket_selected_indices):
    pocket_chains = set()
    pocket_domains = set()
    pocket_linker_count = 0
    for idx in pocket_selected_indices:
        res = subset_residues[idx]
        pocket_chains.add(res['chain'])
        if res['pfam_domain'] == 'Linker':
            pocket_linker_count += 1
        else:
            pocket_domains.add(res['pfam_domain'])

    if len(pocket_chains) == 1 and len(pocket_domains) <= 1 and pocket_linker_count == 0:
        motordock_type = 'Single-Domain'
    elif len(pocket_chains) == 1 and len(pocket_domains) >= 2:
        motordock_type = 'Intra-Chain-Domain-Interface'
    elif len(pocket_chains) == 1 and pocket_linker_count > 0:
        motordock_type = 'Linker'
    elif len(pocket_chains) >= 2 and len(pocket_domains) <= 1:
        motordock_type = 'Chain-Interface'
    elif len(pocket_chains) >= 2 and len(pocket_domains) >= 2:
        motordock_type = 'Mixed-Domain-Chain'
    else:
        motordock_type = 'Unknown'

    return motordock_type, pocket_chains, pocket_domains, pocket_linker_count

def save_pocket_info(path, pdb_id, pocket_file, pocket_data, selected_reskeys, pocket_selected_indices, pocket_mask_tensor, crop_source):
    pocket_center_ca = pocket_data['pocket_center_ca']
    pocket_center_atom = pocket_data['pocket_center_atom']
    payload = {
        'pdb_id': pdb_id,
        'pocket_file': pocket_file,
        'pocket_center_ca': torch.tensor(pocket_center_ca, dtype=torch.float32) if pocket_center_ca is not None else None,
        'pocket_center_atom': torch.tensor(pocket_center_atom, dtype=torch.float32) if pocket_center_atom is not None else None,
        'pocket_ca_coord': torch.tensor(pocket_data['pocket_ca_coords'], dtype=torch.float32),
        'pocket_atom_coord': torch.tensor(pocket_data['pocket_atom_coords'], dtype=torch.float32),
        'pocket_reskeys': list(pocket_data['pocket_reskeys']),
        'selected_reskeys': selected_reskeys,
        'pocket_selected_indices': torch.tensor(pocket_selected_indices, dtype=torch.long),
        'pocket_mask': pocket_mask_tensor,
        'crop_source': crop_source,
    }
    torch.save(payload, path)

def save_domain_frames(path, frames):
    torch.save(frames, path)

def save_candidate_pairs(path, candidate_pairs):
    torch.save(candidate_pairs, path)

def process_single_entry(row, args, pfam_subset):
    pdb_id = row['pdb_id']
    domains = format_pfam_domains(pfam_subset)
    if domains.empty:
        return {'status': 'skip', 'pdb_id': pdb_id, 'reason': 'empty_pfam'}

    domain_lookup = {}
    for d in domains.itertuples():
        domain_lookup.setdefault(d.chain_id, []).append((d.start, d.end, d.pfam_id))

    refined_path = os.path.join(args.pdbbind_root, 'refined-set', pdb_id)
    general_path = os.path.join(args.pdbbind_root, 'v2020-other-PL', pdb_id)
    target_path = general_path if os.path.exists(general_path) else refined_path
    if not os.path.exists(target_path):
        return {'status': 'skip', 'pdb_id': pdb_id, 'reason': 'missing_target_path'}

    prot_file = os.path.join(target_path, f'{pdb_id}_protein.pdb')
    lig_file = os.path.join(target_path, f'{pdb_id}_ligand.mol2')
    pocket_file = os.path.join(target_path, f'{pdb_id}_pocket.pdb')

    if not os.path.exists(prot_file) or not os.path.exists(lig_file):
        print(f'Missing protein or ligand file for {pdb_id}: {prot_file}, {lig_file}')
        return {'status': 'skip', 'pdb_id': pdb_id, 'reason': 'missing_protein_or_ligand'}

    if args.require_pocket and not os.path.exists(pocket_file):
        print(f'Missing pocket file for {pdb_id}: {pocket_file}')
        return {'status': 'skip', 'pdb_id': pdb_id, 'reason': 'missing_pocket'}

    lig = Chem.MolFromMol2File(lig_file, sanitize=False)
    if lig is None:
        print(f'Failed to load ligand for {pdb_id}')
        return {'status': 'skip', 'pdb_id': pdb_id, 'reason': 'ligand_parse_failure'}
    conf = lig.GetConformer()
    lig_coords = np.array([conf.GetAtomPosition(i) for i in range(lig.GetNumAtoms())])
    smiles = Chem.MolToSmiles(lig, canonical=True)

    pocket_data = {
        'pocket_reskeys': set(),
        'pocket_ca_coords': np.zeros((0, 3), dtype=np.float32),
        'pocket_atom_coords': np.zeros((0, 3), dtype=np.float32),
        'pocket_center_ca': None,
        'pocket_center_atom': None,
    }
    if os.path.exists(pocket_file):
        try:
            pocket_data = parse_pocket_pdb(pocket_file)
        except Exception:
            print(f'Failed to parse pocket for {pdb_id}: {pocket_file}')
            if args.require_pocket:
                return {'status': 'skip', 'pdb_id': pdb_id, 'reason': 'pocket_parse_failure'}

    parser_local = PDBParser(QUIET=True)
    try:
        structure = parser_local.get_structure(pdb_id, prot_file)
    except Exception:
        print(f'Failed to parse PDB structure for {pdb_id}')
        return {'status': 'skip', 'pdb_id': pdb_id, 'reason': 'protein_parse_failure'}

    model = next(iter(structure))
    all_residues = []
    for chain in model:
        c_id = chain.get_id()
        current_chain_domains = domain_lookup.get(c_id, [])
        for residue in chain:
            if 'CA' not in residue:
                continue
            if residue.get_resname() not in COMMON_AAS:
                continue
            res_num = residue.get_id()[1]
            icode = residue.get_id()[2].strip()
            assigned_domain = 'Linker'
            cid_domain = f'{c_id}_Linker'
            for d_start, d_end, d_id in current_chain_domains:
                if d_start <= res_num <= d_end:
                    assigned_domain = d_id
                    cid_domain = f'{c_id}_{d_id}'
                    break
            all_residues.append({
                'chain': c_id,
                'resid': int(res_num),
                'icode': icode,
                'resname': residue.get_resname(),
                'coord': residue['CA'].get_coord(),
                'pfam_domain': assigned_domain,
                'cid_domain': cid_domain,
                'reskey': get_residue_key(c_id, residue),
                'obj': residue,
            })

    if not all_residues:
        return {'status': 'skip', 'pdb_id': pdb_id, 'reason': 'empty_protein_residues'}

    prot_coords = np.array([r['coord'] for r in all_residues])
    dists_to_ligand = distance.cdist(prot_coords, lig_coords, 'euclidean').min(axis=1)

    crop_center = None
    if args.crop_source == 'pocket':
        if pocket_data['pocket_center_ca'] is not None:
            crop_center = pocket_data['pocket_center_ca']
        elif pocket_data['pocket_center_atom'] is not None:
            crop_center = pocket_data['pocket_center_atom']

    if len(all_residues) > args.max_seq_len:
        if args.crop_source == 'pocket' and crop_center is not None:
            dists_to_crop_center = np.linalg.norm(prot_coords - crop_center, axis=1)
            sorted_indices = np.argsort(dists_to_crop_center)
        else:
            dists_to_crop_center = dists_to_ligand
            sorted_indices = np.argsort(dists_to_ligand)
        selected_indices = sorted(sorted_indices[:args.max_seq_len])
    else:
        dists_to_crop_center = np.linalg.norm(prot_coords - crop_center, axis=1) if crop_center is not None else dists_to_ligand
        selected_indices = list(range(len(all_residues)))

    if len(selected_indices) < 10:
        print(f'Selected residue count < 10 for {pdb_id}, skipping.')
        return {'status': 'skip', 'pdb_id': pdb_id, 'reason': 'too_few_selected_residues'}

    subset_residues = [all_residues[i] for i in selected_indices]
    subset_dists = [dists_to_ligand[i] for i in selected_indices]

    # Legacy ligand-contact classification retained for backward compatibility.
    subset_domains = set()
    subset_contacting_domains = set()
    for i, res in enumerate(subset_residues):
        d_name = res['pfam_domain']
        if d_name != 'Linker':
            subset_domains.add(d_name)
            if subset_dists[i] <= args.contact_threshold:
                subset_contacting_domains.add(d_name)

    unique_domains_list = sorted(list(subset_domains))
    domain_to_int = {name: idx + 2 for idx, name in enumerate(unique_domains_list)}
    domain_mask_list = []
    for res in subset_residues:
        d_name = res['pfam_domain']
        domain_mask_list.append(1 if d_name == 'Linker' else domain_to_int[d_name])

    n_contacts = len(subset_contacting_domains)
    n_total = len(subset_domains)
    if n_contacts >= 2:
        classification = 'Interface-Binder'
    elif n_contacts == 1:
        classification = 'Single-Domain-Binder'
    elif n_contacts == 0:
        classification = 'Linker-Binder'
    else:
        classification = 'Unknown'

    pocket_reskeys = pocket_data['pocket_reskeys']
    selected_reskeys, pocket_selected_indices, pocket_mask_list, pocket_mask_tensor = match_pocket_to_selected_residues(
        subset_residues, pocket_reskeys
    )

    if args.require_pocket and len(pocket_selected_indices) == 0:
        print(f'Pocket required but no selected residues mapped for {pdb_id}, skipping.')
        return {'status': 'skip', 'pdb_id': pdb_id, 'reason': 'empty_pocket_mapping'}

    prot_ca_coord, prot_seq = get_subset_info(selected_indices, all_residues)

    try:
        assert len(prot_seq) == prot_ca_coord.shape[0]
        assert len(domain_mask_list) == len(prot_seq)
        assert len(pocket_mask_list) == len(prot_seq)
    except AssertionError:
        print(f'Inconsistent lengths for {pdb_id}, skipping.')
        return {'status': 'skip', 'pdb_id': pdb_id, 'reason': 'length_mismatch'}

    domain_units = build_domain_units(subset_residues)
    if len(domain_units) == 0:
        print(f'No domain/chain units built for {pdb_id}, skipping.')
        return {'status': 'skip', 'pdb_id': pdb_id, 'reason': 'no_domain_units'}

    pocket_center = pocket_data['pocket_center_ca'] if pocket_data['pocket_center_ca'] is not None else pocket_data['pocket_center_atom']
    if pocket_center is None:
        if args.require_pocket:
            print(f'Pocket required but no pocket center available for {pdb_id}, skipping.')
            return {'status': 'skip', 'pdb_id': pdb_id, 'reason': 'missing_pocket_center'}
        pocket_center = prot_coords.mean(axis=0)

    nearby_units, candidate_pairs = enumerate_candidate_pairs(domain_units, pocket_center, args.pocket_radius)

    unit_to_frame = {}
    frames = []
    n_stable_frames = 0
    for unit in domain_units:
        if len(unit['ca_coords']) < args.frame_min_ca_atoms:
            R, t, stable, reason, eigvals = None, None, False, 'too_few_ca_atoms', np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            R, t, stable, reason, eigvals = build_local_frame_for_unit(
                unit,
                pocket_center,
                args.frame_min_residues,
                args.frame_eig_ratio_threshold,
            )
        frame_dict = {
            'unit_id': unit['unit_id'],
            'chain_id': unit['chain_id'],
            'pfam_domain': unit['pfam_domain'],
            'unit_type': unit['unit_type'],
            'residue_indices': unit['residue_indices'],
            'R': torch.tensor(R, dtype=torch.float32) if R is not None else None,
            't': torch.tensor(t, dtype=torch.float32) if t is not None else None,
            'stable': stable,
            'reason': reason,
            'eigvals': torch.tensor(eigvals, dtype=torch.float32),
        }
        frames.append(frame_dict)
        if stable and R is not None and t is not None:
            unit_to_frame[unit['unit_id']] = make_frame_matrix(R, t)
            n_stable_frames += 1

    for pair in candidate_pairs:
        ua, ub = pair['unit_a'], pair['unit_b']
        fa_stable = ua in unit_to_frame
        fb_stable = ub in unit_to_frame
        pair['frame_a_stable'] = fa_stable
        pair['frame_b_stable'] = fb_stable
        if fa_stable and fb_stable:
            T_ab = compute_pair_transform(unit_to_frame[ua], unit_to_frame[ub])
            pair['has_native_transform'] = True
            pair['T_ab_native'] = torch.tensor(T_ab, dtype=torch.float32)
        else:
            pair['has_native_transform'] = False
            pair['T_ab_native'] = torch.eye(4, dtype=torch.float32)

    # MotorDock classification is pocket-based and does not use ligand contacts.
    motordock_type, pocket_chains, pocket_domains, pocket_linker_count = classify_motordock_type(
        subset_residues, pocket_selected_indices
    )

    # Candidate pair enumeration is pocket-centered and avoids ligand-contact oracle selection.
    pair_types = sorted(set([p['pair_type'] for p in candidate_pairs]))

    prot_emb_path = os.path.join(args.output_dir, 'embeddings', f'{pdb_id}_prot_emb.pt')
    torch.save({'ca_coord': prot_ca_coord, 'sequence': prot_seq}, prot_emb_path)

    domain_mask_tensor = torch.tensor(domain_mask_list, dtype=torch.long)
    torch.save(domain_mask_tensor, os.path.join(args.output_dir, 'domain_masks', f'{pdb_id}_domain_mask.pt'))
    torch.save(pocket_mask_tensor, os.path.join(args.output_dir, 'pocket_masks', f'{pdb_id}_pocket_mask.pt'))

    save_pocket_info(
        os.path.join(args.output_dir, 'pocket_info', f'{pdb_id}_pocket_info.pt'),
        pdb_id,
        pocket_file,
        pocket_data,
        selected_reskeys,
        pocket_selected_indices,
        pocket_mask_tensor,
        args.crop_source,
    )
    save_domain_frames(os.path.join(args.output_dir, 'frames', f'{pdb_id}_domain_frames.pt'), frames)
    save_candidate_pairs(os.path.join(args.output_dir, 'candidate_pairs', f'{pdb_id}_candidate_pairs.pt'), candidate_pairs)

    pocket_center_csv = pocket_data['pocket_center_ca'] if pocket_data['pocket_center_ca'] is not None else pocket_data['pocket_center_atom']
    if pocket_center_csv is None:
        pocket_center_csv = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    record = {
        'Assembly_ID': pdb_id,
        'smiles': smiles,
        'type': classification,
        'motordock_type': motordock_type,
        'contacting_domains': n_contacts,
        'total_domains': n_total,
        'pocket_contacting_domains': len(pocket_domains),
        'pocket_total_domains': len(set([r['pfam_domain'] for r in subset_residues if r['pfam_domain'] != 'Linker'])),
        'pocket_contacting_chains': len(pocket_chains),
        'pocket_linker_residues': pocket_linker_count,
        'affinity': row['affinity'],
        'target_sequence': prot_seq,
        'protein_file': prot_file,
        'ligand_file': lig_file,
        'pocket_file': pocket_file,
        'has_pocket': bool(len(pocket_reskeys) > 0),
        'n_residues': len(prot_seq),
        'n_pocket_residues': len(pocket_reskeys),
        'n_pocket_selected_residues': len(pocket_selected_indices),
        'n_domain_units': len(domain_units),
        'n_candidate_pairs': len(candidate_pairs),
        'n_nearby_units': len(nearby_units),
        'candidate_pair_types': ';'.join(pair_types),
        'n_stable_frames': n_stable_frames,
        'pocket_center_x': float(pocket_center_csv[0]),
        'pocket_center_y': float(pocket_center_csv[1]),
        'pocket_center_z': float(pocket_center_csv[2]),
    }
    return {'status': 'ok', 'record': record}

def process(args):
    os.makedirs(os.path.join(args.output_dir, 'domain_masks'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'embeddings'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'pocket_masks'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'pocket_info'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'frames'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'candidate_pairs'), exist_ok=True)

    pfam_lst = pd.read_csv(args.pfam_lst_path, sep='\t')
    pfam_lst = pfam_lst.dropna(subset=['AUTH_PDBRES_START', 'AUTH_PDBRES_END']).copy()
    pfam_groups = {k: v for k, v in pfam_lst.groupby('PDB')}

    index_df = load_index_file(args.index_file)
    total_index_entries = len(index_df)
    if args.end_idx != -1:
        index_df = index_df.iloc[args.start_idx:args.end_idx]

    tasks = []
    for _, row in index_df.iterrows():
        pdb_key = row['pdb_id']
        if pdb_key in pfam_groups:
            tasks.append(delayed(process_single_entry)(row, args, pfam_groups[pdb_key]))

    num_cores = 8
    results_raw = Parallel(n_jobs=num_cores)(tqdm(tasks, total=len(tasks)))

    records = [r['record'] for r in results_raw if r is not None and r.get('status') == 'ok']
    skips = [r for r in results_raw if r is not None and r.get('status') == 'skip']

    out_df = pd.DataFrame(records).drop_duplicates(subset=['Assembly_ID']).reset_index(drop=True)

    if not out_df.empty:
        train_ids = set(load_split_ids(args.train_split))
        val_ids = set(load_split_ids(args.val_split))
        test_ids = set(load_split_ids(args.test_split))
        out_df['split'] = out_df.apply(
            assign_split,
            axis=1,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids
        )

    suffix = f'_{args.start_idx}_{args.end_idx}' if args.end_idx != -1 else ''
    out_path = os.path.join(args.output_dir, f'pdbbind_ft{suffix}.csv')
    out_df.to_csv(out_path, index=False)

    skip_df = pd.DataFrame(skips)
    skip_path = os.path.join(args.output_dir, f'pdbbind_skipped{suffix}.csv')
    skip_df.to_csv(skip_path, index=False)

    reason_counts = skip_df['reason'].value_counts().to_dict() if 'reason' in skip_df.columns else {}
    motordock_counts = out_df['motordock_type'].value_counts().to_dict() if 'motordock_type' in out_df.columns else {}

    split_counts = {}
    if 'split' in out_df.columns:
        split_counts = out_df['split'].value_counts().to_dict()

    summary = {
        'total_index_entries': total_index_entries,
        'entries_with_pfam': len(tasks),
        'successfully_processed': len(out_df),
        'skipped_total': len(skip_df),
        'skipped_missing_protein_or_ligand': reason_counts.get('missing_protein_or_ligand', 0),
        'skipped_missing_pocket': reason_counts.get('missing_pocket', 0),
        'skipped_ligand_parse_failure': reason_counts.get('ligand_parse_failure', 0),
        'skipped_protein_parse_failure': reason_counts.get('protein_parse_failure', 0),
        'skipped_empty_pocket_mapping': reason_counts.get('empty_pocket_mapping', 0),
        'counts_by_motordock_type': motordock_counts,
        'counts_by_split': split_counts,
        'skip_reason_counts': reason_counts,
    }
    summary_path = os.path.join(args.output_dir, f'pdbbind_processing_summary{suffix}.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f'Done. Saved {len(out_df)} entries to {out_path}')
    print(f'Saved skipped table to {skip_path}')
    print(f'Saved summary to {summary_path}')


if __name__ == '__main__':
    process(args)