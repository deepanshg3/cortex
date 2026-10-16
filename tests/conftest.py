"""Shared pytest fixtures for the Cortex test suite.

Fast unit tests use synthetic data and do not touch the network. Tests that
build the real MoLFormer/ESM-2 encoders are marked ``slow`` and ``gpu`` and can
be deselected with ``-m "not slow"``.
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import Config  # noqa: E402


@pytest.fixture
def cfg():
    """A small, CPU/AMP-disabled config for fast tests."""
    return Config(
        device="cpu",
        amp_dtype="fp32",
        num_workers=0,
        batch_size=2,
        eval_batch_size=2,
        epochs=1,
        normalize_target=False,
    )


@pytest.fixture
def tiny_parquet(tmp_path):
    """Write a tiny, schema-complete Parquet split and return its path."""
    import polars as pl

    n = 12
    data = {
        "Drug_ID": [f"D{i}" for i in range(n)],
        "Drug": ["CCO", "c1ccccc1", "CC(=O)O", "C", "CCN", "CCCl",
                 "CO", "CCOC", "CC(C)C", "CC(=O)N", "c1ccncc1", "CCCC"] * 1,
        "Target_ID": [f"T{i % 4}" for i in range(n)],
        "Target": ["MKTAYIAKQRQISFVKSHFSRQ"] * n,
        "Y": [10.0, 100.0, 1000.0, 1.0, 5.0, 50.0,
              500.0, 2.0, 20.0, 200.0, 2000.0, 0.5],
        "Drug_Length": [3, 6, 5, 1, 3, 4, 2, 4, 5, 5, 6, 4],
        "Target_Length": [21] * n,
    }
    data["pkd"] = [9.0 - __import__("math").log10(y) for y in data["Y"]]
    path = tmp_path / "tiny.parquet"
    pl.DataFrame(data).write_parquet(path)
    return str(path)


@pytest.fixture
def tiny_parquets(tmp_path, tiny_parquet):
    """train/val/test parquet files with the canonical filenames."""
    import shutil

    out = {}
    for name in ("train", "val", "test"):
        dst = tmp_path / f"{name}.parquet"
        shutil.copy(tiny_parquet, dst)
        out[name] = str(dst)
    return out
