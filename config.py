"""Central configuration for the Cortex DTI project.

Single source of truth for model, data, optimisation and runtime settings.

Design notes
------------
* ``Config`` keeps class-level defaults so that legacy imports
  (``from config import Config; Config.hidden_dim``) continue to work and so
  that the model/tokenizer modules can fall back to sensible defaults.
* A concrete experiment should use :func:`load_config`, which builds an
  *instance* of :class:`Config`, applies explicit overrides and can be
  serialised/hashed for reproducibility and checkpoint compatibility checks.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class Config:
    # ------------------------------------------------------------------
    # 1. Global / experiment identity
    # ------------------------------------------------------------------
    project_name: str = "cortex-dti"
    experiment_name: str = "exp"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # 2. Data & paths
    # ------------------------------------------------------------------
    train_path: str = "data/processed/train.parquet"
    val_path: str = "data/processed/val.parquet"
    test_path: str = "data/processed/test.parquet"

    max_drug_len: int = 200
    max_target_len: int = 1000

    # Expected processed-data schema (used for integrity checks).
    drug_col: str = "Drug"
    target_col: str = "Target"
    label_col: str = "pkd"

    # ------------------------------------------------------------------
    # 3. Model architecture (Hugging Face)
    # ------------------------------------------------------------------
    drug_model_name: str = "ibm/MoLFormer-XL-both-10pct"
    target_model_name: str = "facebook/esm2_t6_8M_UR50D"

    hidden_dim: int = 256
    num_attention_heads: int = 8
    dropout_rate: float = 0.1
    # Compressor downsample factor (kernel == stride).
    compression_factor: int = 4
    # Pooling over the (compressed) protein positions before the MLP head.
    #   "masked" - mean over real, non-padded positions only (correct; default)
    #   "mean"   - mean over all positions including padding (original
    #              behaviour; use only to reproduce legacy checkpoints)
    pooling: str = "masked"
    # Fusion between the drug and target representations.
    #   "cross_attention" - protein queries attend to the drug (default)
    #   "concat"          - pooled drug + pooled target concatenated (simple
    #                       baseline for ablation; doubles the head input width)
    fusion: str = "cross_attention"

    # ------------------------------------------------------------------
    # 4. LoRA settings
    # ------------------------------------------------------------------
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Tuple[str, ...] = ("query", "key", "value", "dense")
    # Encoder fine-tuning policy: "lora" (LoRA adapters only, base frozen) or
    # "full" (unfreeze whole encoder). "lora" is strongly recommended on T4.
    finetune_mode: str = "lora"
    # Optional gradual unfreezing: number of initial epochs to keep the drug /
    # target encoders fully frozen (LoRA also disabled). 0 = train LoRA from
    # the start. See docs/TRAINING_METHODS.md.
    freeze_encoder_epochs: int = 0

    # ------------------------------------------------------------------
    # 5. Training hyperparameters
    # ------------------------------------------------------------------
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    early_stopping_patience: int = 3
    # Gradient accumulation. Effective batch = batch_size * grad_accum_steps.
    grad_accum_steps: int = 1
    grad_clip_norm: float = 1.0

    # Optimiser
    optimizer_name: str = "adamw"  # adamw | adam
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    # Scheduler: linear warmup followed by cosine decay to eta_min.
    scheduler_name: str = "cosine"  # cosine | none
    warmup_ratio: float = 0.05  # fraction of total *steps* used for warmup
    warmup_steps: int = -1  # if >= 0, overrides warmup_ratio
    eta_min: float = 1e-6

    # Loss
    loss_name: str = "mse"  # mse | huber | smoothl1
    huber_delta: float = 1.0

    # Model selection / early stopping. Decoupled from the loss so that runs
    # with different losses (e.g. Huber) are compared on one common yardstick.
    # One of: "val_loss", "rmse", "mae", "pearson", "spearman", "r2".
    # Lower is better for val_loss/rmse/mae; higher is better for the rest.
    select_metric: str = "val_loss"

    # Target normalisation. pKd is standardised using train statistics; the
    # inverse transform is always applied before metrics are reported.
    normalize_target: bool = False
    target_mean: float = 0.0
    target_std: float = 1.0

    # ------------------------------------------------------------------
    # 6. Mixed precision / numerical stability
    # ------------------------------------------------------------------
    # "fp32" | "fp16" | "bf16". On Kaggle T4 only fp32/fp16 are usable.
    amp_dtype: str = "fp16"
    # GradScaler initial loss scale. 65536 (the PyTorch default) exceeds the
    # fp16 max finite value of 65504, so the scale gradient flowing into fp16
    # autocast nodes overflows to inf and poisons every parameter gradient.
    # 4096 keeps it safely inside fp16 range while still preventing underflow;
    # the scaler then adapts it automatically.
    amp_init_scale: float = 4096.0
    # MoLFormer's custom linear attention overflows in fp16. Its tower is
    # therefore executed in fp32 even when AMP is enabled. See docs.
    drug_tower_fp32: bool = True

    # ------------------------------------------------------------------
    # 7. Online hard-example mining (OHEM)
    # ------------------------------------------------------------------
    hard_mining_start_epoch: int = 5
    hard_mining_fraction: float = 0.5

    # ------------------------------------------------------------------
    # 8. DataLoader
    # ------------------------------------------------------------------
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True

    # Determinism (cuDNN). Slower; leave off unless bit-reproducibility is
    # required (the split generation is already deterministic without this).
    deterministic: bool = False

    # Provenance of the data split protocol (informational; set by the
    # preprocessing pipeline: cold_target / cold_drug / cold_pair / random).
    data_protocol: str = "unknown"
    split_version: str = "unknown"

    # ------------------------------------------------------------------
    # 9. Checkpointing / logging
    # ------------------------------------------------------------------
    output_dir: str = "experiments"
    checkpoint_dir: Optional[str] = None  # defaults to <output_dir>/<experiment>
    save_every_steps: int = 0  # 0 = epoch boundaries only
    save_best: bool = True
    save_latest: bool = True
    keep_last_n: int = 3  # number of periodic step checkpoints to retain
    log_every_steps: int = 50
    eval_every_epochs: int = 1

    # Inference / evaluation
    inference_weights_path: str = "checkpoints/best_model.pt"
    use_fp16_inference: bool = True
    eval_batch_size: int = 64

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @classmethod
    def field_names(cls):
        return [f.name for f in fields(cls)]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def config_hash(self, ignore=("device", "experiment_name", "output_dir",
                                  "checkpoint_dir", "num_workers",
                                  "target_mean", "target_std")) -> str:
        """Stable hash of the *scientific* configuration.

        Volatile/runtime-only fields (device, paths that depend on the machine)
        are ignored so that a checkpoint can be validated across machines.
        """
        d = {k: v for k, v in self.to_dict().items() if k not in ignore}
        # Tuples -> lists for stable JSON.
        payload = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def resolved_checkpoint_dir(self) -> str:
        return self.checkpoint_dir or os.path.join(self.output_dir, self.experiment_name)

    def resolved_torch_dtype(self):
        return {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.amp_dtype]

    def uses_amp(self) -> bool:
        return self.amp_dtype in ("fp16", "bf16") and self.device.startswith("cuda")


def apply_overrides(cfg: Config, overrides: Dict[str, Any]) -> Config:
    """Apply ``{"field": value}`` overrides, rejecting unknown fields."""
    valid = set(cfg.field_names())
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(f"Unknown configuration override(s): {sorted(unknown)}")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def load_config(**overrides) -> Config:
    """Build a fresh :class:`Config` instance, applying overrides."""
    return apply_overrides(Config(), overrides)
