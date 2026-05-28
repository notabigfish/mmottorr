from pathlib import Path
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from motordock.data.ablation_dataset import RepresentationAblationDataset


def test_ablation_no_oracle_selection(tmp_path: Path):
    out = tmp_path / "data"
    for d in ["embeddings","domain_masks","pocket_masks","pocket_info","frames","candidate_pairs"]:
        (out/d).mkdir(parents=True)
    pid = "x"
    torch.save({"ca_coord": torch.randn(8,3), "sequence": "ACDEFGHI"}, out/"embeddings"/f"{pid}_prot_emb.pt")
    torch.save(torch.ones(8,dtype=torch.long), out/"domain_masks"/f"{pid}_domain_mask.pt")
    torch.save(torch.ones(8,dtype=torch.long), out/"pocket_masks"/f"{pid}_pocket_mask.pt")
    torch.save({"pocket_center_ca": torch.zeros(3), "pocket_center_atom": None}, out/"pocket_info"/f"{pid}_pocket_info.pt")
    torch.save([{"unit_id":"A","stable":True,"R":torch.eye(3),"t":torch.zeros(3),"residue_indices":[0]}], out/"frames"/f"{pid}_domain_frames.pt")
    torch.save([{"pair_id":"A__A","unit_a":"A","unit_b":"A","pair_type":"single","unit_a_type":"pfam_domain","unit_b_type":"pfam_domain","unit_a_chain":"A","unit_b_chain":"A","unit_a_num_residues":1,"unit_b_num_residues":1,"frame_a_stable":True,"frame_b_stable":True,"has_native_transform":True,"T_ab_native":torch.eye(4)}], out/"candidate_pairs"/f"{pid}_candidate_pairs.pt")
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO")); AllChem.EmbedMolecule(mol, randomSeed=1)
    sdf = tmp_path/"l.sdf"; w = Chem.SDWriter(str(sdf)); w.write(mol); w.close()
    pd.DataFrame([{"Assembly_ID":pid,"split":"train","motordock_type":"Single-Domain","protein_file":"p","ligand_file":str(sdf),"pocket_file":"k","affinity":0.0}]).to_csv(tmp_path/"ft.csv", index=False)
    ds = RepresentationAblationDataset(str(tmp_path/"ft.csv"), str(out), split="train", randomize_pose=False, perturb_pair_transform=False, representation="se3_log")
    s = ds[0]
    assert "pair_features" in s
