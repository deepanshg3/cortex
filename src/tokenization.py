"""Dataset and dynamic-padding collator for Cortex.

Reads the DVC-tracked Parquet splits, tokenises SMILES with the MoLFormer
tokenizer and protein FASTA with the ESM-2 tokenizer, and returns per-sample
tensors. Padding is applied to ``max_*_len`` here; the collator trims to the
batch maximum to avoid wasted compute.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config  # noqa: E402
from src.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def _read_parquet(path: str):
    """Read parquet preferring polars, falling back to pandas."""
    try:
        import polars as pl

        return pl.read_parquet(path), "polars"
    except Exception:
        import pandas as pd

        return pd.read_parquet(path), "pandas"


class BindingDataset(Dataset):
    def __init__(self, data_path: str, config: Config | None = None):
        self.config = config if config is not None else Config()
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path not found: {data_path}")

        df, backend = _read_parquet(data_path)
        self._backend = backend

        required = [self.config.drug_col, self.config.target_col, self.config.label_col]
        cols = df.columns
        missing = [c for c in required if c not in cols]
        if missing:
            raise ValueError(f"{data_path} missing required columns: {missing}")

        if backend == "polars":
            self.drugs = df.get_column(self.config.drug_col).to_numpy()
            self.targets = df.get_column(self.config.target_col).to_numpy()
            self.labels = df.get_column(self.config.label_col).to_numpy().astype("float64")
            self.drug_ids = (
                df.get_column("Drug_ID").to_numpy() if "Drug_ID" in cols else None
            )
            self.target_ids = (
                df.get_column("Target_ID").to_numpy() if "Target_ID" in cols else None
            )
        else:
            self.drugs = df[self.config.drug_col].to_numpy()
            self.targets = df[self.config.target_col].to_numpy()
            self.labels = df[self.config.label_col].to_numpy(dtype="float64")
            self.drug_ids = df["Drug_ID"].to_numpy() if "Drug_ID" in cols else None
            self.target_ids = df["Target_ID"].to_numpy() if "Target_ID" in cols else None

        logger.info(
            "Initializing tokenizers (drug=%s, target=%s)",
            self.config.drug_model_name,
            self.config.target_model_name,
        )
        self.drug_tokenizer = AutoTokenizer.from_pretrained(
            self.config.drug_model_name, trust_remote_code=True
        )
        self.target_tokenizer = AutoTokenizer.from_pretrained(self.config.target_model_name)

    def __len__(self) -> int:
        return len(self.labels)

    def set_label_normalization(self, mean: float, std: float) -> None:
        """Enable standardisation of the label returned by ``__getitem__``."""
        self._label_mean = float(mean)
        self._label_std = float(std) if std else 1.0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        drug_str = str(self.drugs[idx])
        target_str = str(self.targets[idx])
        label = float(self.labels[idx])
        if getattr(self, "_label_mean", None) is not None:
            label = (label - self._label_mean) / self._label_std

        drug_tokens = self.drug_tokenizer(
            drug_str,
            max_length=self.config.max_drug_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_tokens = self.target_tokenizer(
            target_str,
            max_length=self.config.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "drug_input_ids": drug_tokens["input_ids"].squeeze(0),
            "drug_attention_mask": drug_tokens["attention_mask"].squeeze(0),
            "target_input_ids": target_tokens["input_ids"].squeeze(0),
            "target_attention_mask": target_tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float32),
        }


def dynamic_collate_fn(batch: List[Dict[str, torch.Tensor]]):
    """Trim padded sequences to the batch maximum, then default-collate."""
    max_drug = max(int((item["drug_attention_mask"] == 1).sum().item()) for item in batch)
    max_target = max(int((item["target_attention_mask"] == 1).sum().item()) for item in batch)
    for item in batch:
        item["drug_input_ids"] = item["drug_input_ids"][:max_drug]
        item["drug_attention_mask"] = item["drug_attention_mask"][:max_drug]
        item["target_input_ids"] = item["target_input_ids"][:max_target]
        item["target_attention_mask"] = item["target_attention_mask"][:max_target]
    return torch.utils.data.dataloader.default_collate(batch)
