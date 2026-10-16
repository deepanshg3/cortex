"""Data integrity / leakage audit tests (src.data_audit)."""

from __future__ import annotations

import math

import polars as pl

from src.data_audit import integrity_report, leakage_report, run


def test_pkd_matches_formula(tiny_parquet):
    df = pl.read_parquet(tiny_parquet).to_pandas()
    rep = integrity_report(df, "tiny")
    assert rep["pkd_matches_9_minus_log10_Y"] is True
    assert rep["n_nonfinite_labels"] == 0
    assert rep["rows"] == 12


def test_leakage_detects_self_overlap(tiny_parquets):
    tr = pl.read_parquet(tiny_parquets["train"]).to_pandas()
    va = pl.read_parquet(tiny_parquets["val"]).to_pandas()
    te = pl.read_parquet(tiny_parquets["test"]).to_pandas()
    rep = leakage_report(tr, va, te)
    # All three splits point at the same tiny data -> full overlap.
    assert rep["pair_overlap_train_val"] == len(tr)
    assert rep["target_overlap_train_test"] == tr["Target"].nunique()
    assert rep["drug_overlap_train_test"] == tr["Drug"].nunique()


def test_run_reports_all_splits(tiny_parquets, tmp_path):
    # run() expects data_dir/{train,val,test}.parquet naming.
    report = run(str(tmp_path))
    assert [r["split"] for r in report["integrity"]] == ["train", "val", "test"]
    assert set(report["leakage"]) >= {"pair_overlap_train_test", "n_train"}


def test_integrity_flags_duplicates():
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(
        {
            "Drug": ["CCO", "CCO", "CCN"],
            "Target": ["MKT", "MKT", "MKT"],
            "pkd": [5.0, 5.0, 6.0],
            "Y": [10000.0, 10000.0, 1000.0],
        }
    )
    rep = integrity_report(df, "dup")
    assert rep["n_dup_pairs"] == 1
    assert rep["n_dup_full_rows"] == 1
    assert rep["n_unique_drugs"] == 2
    assert rep["n_unique_targets"] == 1
    assert math.isclose(rep["label_mean"], np.mean([5.0, 5.0, 6.0]))
