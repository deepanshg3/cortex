"""Data integrity and leakage auditing for Cortex.

Read-only helpers that inspect the processed Parquet splits. They never modify
the dataset. Used by ``tests/`` and can be run directly:

    python -m src.data_audit --data-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _read(path):
    try:
        import polars as pl

        return pl.read_parquet(path).to_pandas()
    except Exception:
        import pandas as pd

        return pd.read_parquet(path)


def integrity_report(df, name: str, label_col: str = "pkd", drug_col="Drug", target_col="Target") -> Dict:
    n = len(df)
    labels = df[label_col].to_numpy(dtype="float64")
    report = {
        "split": name,
        "rows": int(n),
        "cols": list(df.columns),
        "label_col": label_col,
        "label_min": float(np.nanmin(labels)),
        "label_max": float(np.nanmax(labels)),
        "label_mean": float(np.nanmean(labels)),
        "label_std": float(np.nanstd(labels)),
        "n_nonfinite_labels": int((~np.isfinite(labels)).sum()),
        "n_null_drug": int(df[drug_col].isna().sum()),
        "n_null_target": int(df[target_col].isna().sum()),
        "n_empty_drug": int((df[drug_col].astype(str).str.len() == 0).sum()),
        "n_empty_target": int((df[target_col].astype(str).str.len() == 0).sum()),
        "n_dup_pairs": int(df.duplicated([drug_col, target_col]).sum()),
        "n_dup_full_rows": int(df.duplicated([drug_col, target_col, label_col]).sum()),
        "n_unique_drugs": int(df[drug_col].nunique()),
        "n_unique_targets": int(df[target_col].nunique()),
    }
    if "Y" in df.columns:
        y = df["Y"].to_numpy(dtype="float64")
        pkd = df[label_col].to_numpy(dtype="float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = 9.0 - np.log10(np.where(y > 0, y, np.nan))
        finite = np.isfinite(expected)
        report["pkd_matches_9_minus_log10_Y"] = bool(
            np.allclose(expected[finite], pkd[finite], atol=1e-4)
        )
        report["n_Y_le_zero"] = int((y <= 0).sum())
    return report


def leakage_report(tr, va, te, drug_col="Drug", target_col="Target") -> Dict:
    def pairs(d):
        return set(zip(d[drug_col].astype(str), d[target_col].astype(str)))

    def uniq(d, c):
        return set(d[c].astype(str))

    pt, pv, pe = pairs(tr), pairs(va), pairs(te)
    return {
        "pair_overlap_train_val": len(pt & pv),
        "pair_overlap_train_test": len(pt & pe),
        "pair_overlap_val_test": len(pv & pe),
        "drug_overlap_train_val": len(uniq(tr, drug_col) & uniq(va, drug_col)),
        "drug_overlap_train_test": len(uniq(tr, drug_col) & uniq(te, drug_col)),
        "target_overlap_train_val": len(uniq(tr, target_col) & uniq(va, target_col)),
        "target_overlap_train_test": len(uniq(tr, target_col) & uniq(te, target_col)),
        "n_train": len(tr),
        "n_val": len(va),
        "n_test": len(te),
    }


def run(data_dir: str, label_col: str = "pkd") -> Dict:
    tr = _read(os.path.join(data_dir, "train.parquet"))
    va = _read(os.path.join(data_dir, "val.parquet"))
    te = _read(os.path.join(data_dir, "test.parquet"))
    report = {
        "integrity": [
            integrity_report(tr, "train", label_col),
            integrity_report(va, "val", label_col),
            integrity_report(te, "test", label_col),
        ],
        "leakage": leakage_report(tr, va, te),
    }
    return report


def main():
    ap = argparse.ArgumentParser(description="Cortex data integrity/leakage audit")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--label-col", default="pkd")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    report = run(args.data_dir, args.label_col)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        for r in report["integrity"]:
            print(f"--- {r['split']} ---")
            for k, v in r.items():
                if k != "split":
                    print(f"  {k}: {v}")
        print("--- leakage ---")
        for k, v in report["leakage"].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
