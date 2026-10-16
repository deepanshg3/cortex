"""Dataset, collator and label-normalisation tests (src.tokenization)."""

from __future__ import annotations

import math

import polars as pl
import pytest
import torch

from config import Config
from src.tokenization import BindingDataset, dynamic_collate_fn


def test_missing_columns_raise(tmp_path):
    path = tmp_path / "bad.parquet"
    pl.DataFrame({"Drug": ["CCO"], "Target": ["MKT"]}).write_parquet(path)
    with pytest.raises(ValueError, match="missing required columns"):
        BindingDataset(str(path), Config())


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        BindingDataset("/no/such/file.parquet", Config())


def test_dataset_len_and_shapes(tiny_parquet, cfg):
    ds = BindingDataset(tiny_parquet, cfg)
    assert len(ds) == 12
    item = ds[0]
    assert item["drug_input_ids"].shape == (cfg.max_drug_len,)
    assert item["target_input_ids"].shape == (cfg.max_target_len,)
    assert item["label"].dtype == torch.float32


def test_label_normalization_roundtrip(tiny_parquet, cfg):
    ds = BindingDataset(tiny_parquet, cfg)
    raw = float(ds[3]["label"])
    mean, std = 6.0, 1.5
    ds.set_label_normalization(mean, std)
    norm = float(ds[3]["label"])
    assert norm == pytest.approx((raw - mean) / std, abs=1e-6)


def test_collate_trims_to_batch_max(tiny_parquet, cfg):
    ds = BindingDataset(tiny_parquet, cfg)
    batch = [ds[0], ds[1], ds[2]]
    # Valid (unpadded) lengths per sample.
    valid_drug = [int((ds[i]["drug_attention_mask"] == 1).sum()) for i in range(3)]
    valid_target = [int((ds[i]["target_attention_mask"] == 1).sum()) for i in range(3)]
    out = dynamic_collate_fn(batch)
    assert out["drug_input_ids"].shape[1] == max(valid_drug)
    assert out["target_input_ids"].shape[1] == max(valid_target)
    # No sequence is longer than the batch max.
    assert out["drug_input_ids"].shape[1] <= cfg.max_drug_len


def test_pkd_formula_alignment(tiny_parquet, cfg):
    ds = BindingDataset(tiny_parquet, cfg)
    import polars as pl

    df = pl.read_parquet(tiny_parquet)
    y = df["Y"].to_numpy()
    pkd = df["pkd"].to_numpy()
    for i in (0, 3, 11):
        if y[i] > 0:
            assert pkd[i] == pytest.approx(9.0 - math.log10(y[i]), abs=1e-6)
        assert float(ds[i]["label"]) == pytest.approx(pkd[i], abs=1e-6)
