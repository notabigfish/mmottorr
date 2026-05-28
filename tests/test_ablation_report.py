from pathlib import Path
import pandas as pd
import numpy as np
from motordock.eval.statistical_tests import bootstrap_paired_delta_ci
from motordock.eval.ablation_report import summarize_run


def test_bootstrap_paired_delta_ci():
    x = np.array([0,1,1,0,1], dtype=float)
    y = np.array([1,1,1,0,1], dtype=float)
    d, lo, hi = bootstrap_paired_delta_ci(x,y)
    assert lo <= d <= hi


def test_summary_report_reads_prediction_csvs(tmp_path: Path):
    run = tmp_path/"runs"/"se3_log"
    run.mkdir(parents=True)
    pd.DataFrame([{"pdb_id":"a","confidence":1.0,"rmsd":1.2,"success_2A":1,"centroid_distance":1.0,"motordock_type":"Single-Domain"}]).to_csv(run/"val_predictions.csv", index=False)
    pd.DataFrame([{"num_parameters":100}]).to_csv(run/"train_log.csv", index=False)
    s = summarize_run(str(tmp_path/"runs"))
    assert len(s) > 0


def test_summary_contains_required_categories(tmp_path: Path):
    run = tmp_path/"runs"/"se3_log"
    run.mkdir(parents=True)
    pd.DataFrame([{"pdb_id":"a","confidence":1.0,"rmsd":1.2,"success_2A":1,"centroid_distance":1.0,"motordock_type":"Single-Domain"}]).to_csv(run/"val_predictions.csv", index=False)
    s = summarize_run(str(tmp_path/"runs"))
    assert "Overall" in set(s["motordock_type"]) 
