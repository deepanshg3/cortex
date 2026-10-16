"""Reproducible, deterministic dataset construction and split generation.

This module builds Cortex training splits from the raw TDC ``BindingDB_Kd``
export. It is deliberately **self-contained and library-version independent**:
splits are assigned by hashing identifiers with SHA-256 (``seed|salt|key``)
rather than by ``DataFrame.sample(seed=...)``, so the exact same train/val/test
partition is produced on any machine, any pandas/polars version.

Design guarantees
-----------------
* The historical ``data/processed/*.parquet`` splits are **never touched**.
  New splits are written under ``--out-dir`` (default ``data/splits/<version>``).
* Duplicate ``(Drug, Target)`` pairs are aggregated *before* splitting, so the
  same pair can never appear in two splits (this removes the 769 train/val pair
  leaks and the duplicate-label noise documented in ``docs/FINAL_REPORT.md``).
* Every split protocol ships a machine-readable manifest with the raw-file hash,
  the exact parameters, per-split counts and a leakage report.

Protocols
---------
``cold_target``
    All pairs sharing a held-out target go to test: no target seen in training
    (the original benchmark's protocol).
``cold_drug``
    Held-out drugs; no drug seen in training.
``cold_pair``
    Both a held-out drug set and a held-out target set; test contains only pairs
    whose drug *and* target are both held out. Guarantees zero drug **and** zero
    target overlap between train and test.
``random``
    Uniform random pair-level split (upper bound; leaks molecules/proteins).

Usage::

    python -m src.preprocess --out-dir data/splits/v1
    python -m src.preprocess --protocols cold_target cold_drug --dedup median
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import data_audit  # noqa: E402
from src.logger import get_logger  # noqa: E402

logger = get_logger("src.preprocess")

SUPPORTED_PROTOCOLS = ("cold_target", "cold_drug", "cold_pair", "random")
DEDUP_POLICIES = ("median", "mean", "first", "min", "max")
PAIR_SEP = "\x1f"


# ----------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------
def _read_raw(path: str):
    import polars as pl

    if path.endswith(".parquet"):
        df = pl.read_parquet(path)
    elif path.endswith(".tab"):
        df = pl.read_csv(path, separator="\t")
    else:
        df = pl.read_csv(path)
    return df


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------
# Label + cleaning
# ----------------------------------------------------------------------
def add_labels(df, eps: float = 1e-10, nonpositive: str = "keep"):
    """Add length columns and the ``pkd = 9 - log10(Y + eps)`` label.

    ``nonpositive="drop"`` removes rows with ``Y <= 0`` (invalid Kd) instead of
    mapping them to the sentinel pkd ~ 19.
    """
    import polars as pl

    df = df.with_columns([
        pl.col("Drug").cast(pl.Utf8).str.len_chars().alias("Drug_Length"),
        pl.col("Target").cast(pl.Utf8).str.len_chars().alias("Target_Length"),
        pl.col("Y").cast(pl.Float64),
    ])
    n_nonpos = int((df["Y"] <= 0).sum())
    if nonpositive == "drop" and n_nonpos:
        df = df.filter(pl.col("Y") > 0)
    df = df.with_columns(
        (9.0 - (pl.col("Y") + eps).log10()).alias("pkd")
    )
    return df, n_nonpos


def filter_lengths(df, max_drug_len: int, max_target_len: int):
    import polars as pl

    return df.filter(
        (pl.col("Drug_Length") <= max_drug_len)
        & (pl.col("Target_Length") <= max_target_len)
    )


# ----------------------------------------------------------------------
# Deduplication
# ----------------------------------------------------------------------
def deduplicate(df, policy: str = "median") -> Tuple[object, Dict]:
    """Aggregate duplicate ``(Drug, Target)`` pairs.

    Aggregation happens on the **pkd (log10 affinity) scale**, which is the
    modeling target; medians therefore commute with the label transform.

    Returns ``(deduped_df, stats)``.
    """
    import polars as pl

    if policy not in DEDUP_POLICIES:
        raise ValueError(f"dedup policy must be one of {DEDUP_POLICIES}, got {policy!r}")

    n_before = df.height
    group = df.group_by(["Drug", "Target"], maintain_order=True).agg([
        pl.col("pkd").n_unique().alias("_nuniq"),
        pl.col("pkd").count().alias("n_meas"),
        pl.col("pkd").std(ddof=0).alias("pkd_std"),
        pl.col("pkd").min().alias("_pmin"),
        pl.col("pkd").max().alias("_pmax"),
        pl.col("Y").min().alias("Y"),
        pl.col("Drug_ID").first().alias("Drug_ID"),
        pl.col("Target_ID").first().alias("Target_ID"),
        pl.col("Drug_Length").first().alias("Drug_Length"),
        pl.col("Target_Length").first().alias("Target_Length"),
        pl.col("pkd").median().alias("_pmedian"),
        pl.col("pkd").mean().alias("_pmean"),
        pl.col("pkd").first().alias("_pfirst"),
    ])

    agg = {
        "median": pl.col("_pmedian"),
        "mean": pl.col("_pmean"),
        "first": pl.col("_pfirst"),
        "min": pl.col("_pmin"),
        "max": pl.col("_pmax"),
    }[policy]

    conflict = group.filter(pl.col("_nuniq") > 1)
    range_span = (conflict["_pmax"] - conflict["_pmin"]) if conflict.height else None

    group = group.with_columns(agg.alias("pkd"))
    # Recompute a consistent Y from the aggregated (log-scale) pkd.
    group = group.with_columns(
        ((10.0 ** (9.0 - pl.col("pkd"))) - 1e-10).alias("Y")
    ).drop(["_pmin", "_pmax", "_pmedian", "_pmean", "_pfirst", "_nuniq"])

    stats = {
        "policy": policy,
        "n_rows_input": int(n_before),
        "n_unique_pairs": int(group.height),
        "n_rows_dropped": int(n_before - group.height),
        "n_duplicate_groups": int(group.filter(pl.col("n_meas") > 1).height),
        "n_conflicting_groups": int(conflict.height),
        "max_label_range_log10_units": (
            float(range_span.max()) if range_span is not None and range_span.len() else 0.0
        ),
    }
    return group, stats


# ----------------------------------------------------------------------
# Deterministic bucketing + splits
# ----------------------------------------------------------------------
def _bucket(key: str, seed: int, salt: str) -> float:
    h = hashlib.sha256(f"{seed}|{salt}|{key}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def _bucket_array(keys, seed: int, salt: str) -> np.ndarray:
    return np.fromiter((_bucket(k, seed, salt) for k in keys), dtype=np.float64, count=len(keys))


def make_splits(df, protocol: str, seed: int, test_frac: float, val_frac: float) -> Dict:
    """Return ``{"train","val","test"}`` boolean masks (numpy) + metadata."""
    import polars as pl

    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}")
    drugs = df["Drug"].to_list()
    targets = df["Target"].to_list()
    pairs = [f"{d}{PAIR_SEP}{t}" for d, t in zip(drugs, targets)]
    n = len(pairs)

    pair_test = _bucket_array(pairs, seed, "test_pair")
    pair_val = _bucket_array(pairs, seed, "val_pair")
    drug_hold = _bucket_array(drugs, seed, "test_drug")
    target_hold = _bucket_array(targets, seed, "test_target")
    d_hold = drug_hold < test_frac
    t_hold = target_hold < test_frac

    if protocol == "random":
        test = pair_test < test_frac
        pool = ~test
    elif protocol == "cold_target":
        test = t_hold
        pool = ~test
    elif protocol == "cold_drug":
        test = d_hold
        pool = ~test
    elif protocol == "cold_pair":
        test = d_hold & t_hold
        pool = (~d_hold) & (~t_hold)
    else:  # pragma: no cover
        raise ValueError(protocol)

    # Val is drawn from the non-test pool. ``val_frac`` is the fraction of that
    # pool; rate normalisation keeps the requested overall proportion.
    val = pool & (pair_val < val_frac)
    train = pool & ~val

    meta = {
        "protocol": protocol,
        "seed": seed,
        "test_frac": test_frac,
        "val_frac_of_pool": val_frac,
        "counts": {
            "train": int(train.sum()),
            "val": int(val.sum()),
            "test": int(test.sum()),
            "excluded": int((~pool & ~test).sum()),
        },
    }
    return {"train": train, "val": val, "test": test, "meta": meta}


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def build(raw_path: str, out_dir: str, *, protocols: List[str], seed: int = 42,
          test_frac: float = 0.10, val_frac: float = 0.15,
          max_drug_len: int = 200, max_target_len: int = 1000,
          dedup: str = "median", eps: float = 1e-10,
          nonpositive: str = "keep", version: str = "v1") -> Dict:
    import polars as pl

    if raw_path.endswith(".csv") and not os.path.exists(raw_path):
        raise FileNotFoundError(raw_path)
    logger.info("Reading raw data: %s", raw_path)
    raw = _read_raw(raw_path)
    raw_sha = sha256_file(raw_path)

    df, n_nonpos = add_labels(raw, eps=eps, nonpositive=nonpositive)
    n_after_label = df.height
    df = filter_lengths(df, max_drug_len, max_target_len)
    n_after_len = df.height
    df, dedup_stats = deduplicate(df, policy=dedup)
    # Canonical ordering => stable parquet bytes and stable audit output.
    df = df.sort(["Drug", "Target"])

    manifest = {
        "version": version,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "raw": {
            "path": raw_path,
            "sha256": raw_sha,
            "rows": int(raw.height),
            "columns": list(raw.columns),
        },
        "params": {
            "seed": seed, "test_frac": test_frac, "val_frac_of_pool": val_frac,
            "max_drug_len": max_drug_len, "max_target_len": max_target_len,
            "dedup": dedup, "eps": eps, "nonpositive": nonpositive,
        },
        "label": "pkd = 9 - log10(Y + eps)  # Y = Kd in nM",
        "cleaning": {
            "n_nonpositive_Y": int(n_nonpos),
            "rows_after_label": int(n_after_label),
            "rows_after_length_filter": int(n_after_len),
        },
        "dedup": dedup_stats,
        "protocols": {},
    }

    os.makedirs(out_dir, exist_ok=True)
    for protocol in protocols:
        split = make_splits(df, protocol, seed, test_frac, val_frac)
        sub = os.path.join(out_dir, protocol)
        os.makedirs(sub, exist_ok=True)
        files = {}
        frames = {}
        for name in ("train", "val", "test"):
            part = df.filter(pl.Series(split[name]))
            files[name] = os.path.join(sub, f"{name}.parquet")
            part.write_parquet(files[name])
            frames[name] = part
        leak = data_audit.leakage_report(
            frames["train"].to_pandas(), frames["val"].to_pandas(), frames["test"].to_pandas()
        )
        integrity = {
            name: data_audit.integrity_report(frames[name].to_pandas(), name)
            for name in ("train", "val", "test")
        }
        entry = {**split["meta"], "files": files, "leakage": leak, "integrity": integrity}
        manifest["protocols"][protocol] = entry
        with open(os.path.join(sub, "split_manifest.json"), "w") as fh:
            json.dump(entry, fh, indent=2, default=str)
        logger.info(
            "Protocol %-12s -> train=%d val=%d test=%d | target_overlap(tr,te)=%d "
            "drug_overlap(tr,te)=%d",
            protocol, entry["counts"]["train"], entry["counts"]["val"],
            entry["counts"]["test"], leak["target_overlap_train_test"],
            leak["drug_overlap_train_test"],
        )

    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    logger.info("Wrote split manifest -> %s", os.path.join(out_dir, "manifest.json"))
    return manifest


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Build reproducible Cortex splits")
    ap.add_argument("--raw", default="data/raw/raw_binding_data.csv")
    ap.add_argument("--out-dir", default="data/splits/v1")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--protocols", nargs="+", default=list(SUPPORTED_PROTOCOLS),
                    choices=list(SUPPORTED_PROTOCOLS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-drug-len", type=int, default=200)
    ap.add_argument("--max-target-len", type=int, default=1000)
    ap.add_argument("--dedup", default="median", choices=list(DEDUP_POLICIES))
    ap.add_argument("--nonpositive", default="keep", choices=["keep", "drop"])
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    build(
        args.raw, args.out_dir, protocols=args.protocols, seed=args.seed,
        test_frac=args.test_frac, val_frac=args.val_frac,
        max_drug_len=args.max_drug_len, max_target_len=args.max_target_len,
        dedup=args.dedup, nonpositive=args.nonpositive, version=args.version,
    )


if __name__ == "__main__":
    main()
