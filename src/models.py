"""Cortex dual-encoder drug-target binding-affinity model.

Architecture (verified against the original implementation and preserved):

    SMILES --> MoLFormer-XL (+LoRA) --> drug_proj (768 -> hidden)
    FASTA  --> ESM-2 8M      (+LoRA) --> target_proj (320 -> hidden)
                                        --> Conv1d compressor (÷4)
    cross-attention (protein queries drug) --> masked mean pool --> MLP --> pKd

Two correctness fixes relative to the original code:

1. ``drug_tower_fp32``: MoLFormer implements *linear* (Performer-style)
   attention with relu random-feature maps and no softmax normaliser. When
   autocast casts its ``torch.matmul`` calls to fp16, the unnormalised
   accumulations overflow the fp16 exponent range (max ~6.5e4), producing
   ``inf`` and then ``nan``. Running the drug tower in fp32 keeps AMP for the
   rest of the network while remaining numerically stable. Verified: AMP-only
   produces NaN at layer 0; the fp32 drug tower produces a finite loss.

2. ``masked mean pooling``: the original code averaged cross-attention outputs
   over *all* protein positions, including padding created by the compressor.
   This dilutes the representation. The masked mean (default, ``pooling="masked"``)
   uses the compressed attention mask. No parameters change, so existing weight
   files still load. ``pooling="mean"`` reproduces the original behaviour and is
   provided to evaluate legacy checkpoints faithfully, since the legacy weights
   were *trained* with that pooling.
"""

from __future__ import annotations

import contextlib
import os
import sys

import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config  # noqa: E402
from src.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


class BindingModel(nn.Module):
    def __init__(self, config: Config | None = None):
        super().__init__()
        cfg = config if config is not None else Config()
        self.config = cfg
        self.drug_tower_fp32 = bool(getattr(cfg, "drug_tower_fp32", True))

        logger.info("Building Tower 1: Chemical Model (MolFormer: %s)", cfg.drug_model_name)
        self.drug_tower = AutoModel.from_pretrained(
            cfg.drug_model_name, deterministic_eval=True, trust_remote_code=True
        )

        logger.info("Building Tower 2: Protein Model (ESM-2: %s)", cfg.target_model_name)
        self.target_tower = AutoModel.from_pretrained(cfg.target_model_name)

        if hasattr(self.drug_tower, "gradient_checkpointing_enable"):
            self.drug_tower.gradient_checkpointing_enable()
        if hasattr(self.target_tower, "gradient_checkpointing_enable"):
            self.target_tower.gradient_checkpointing_enable()

        if cfg.finetune_mode == "lora":
            drug_lora_config = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                target_modules=list(cfg.target_modules),
                lora_dropout=cfg.lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            target_lora_config = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                target_modules=list(cfg.target_modules),
                lora_dropout=cfg.lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            logger.info("Injecting LoRA adapters (r=%d, alpha=%d).", cfg.lora_r, cfg.lora_alpha)
            self.drug_tower = get_peft_model(self.drug_tower, drug_lora_config)
            self.target_tower = get_peft_model(self.target_tower, target_lora_config)
        elif cfg.finetune_mode == "full":
            logger.warning("finetune_mode='full': unfreezing full encoders (memory heavy).")
        elif cfg.finetune_mode == "frozen":
            # Feature-extraction baseline: no LoRA, encoders fully frozen. Only
            # the projections, compressor, fusion and head are trained.
            logger.info("finetune_mode='frozen': encoders frozen (feature extraction).")
            for tower in (self.drug_tower, self.target_tower):
                for p in tower.parameters():
                    p.requires_grad = False
        else:
            raise ValueError(f"Unknown finetune_mode: {cfg.finetune_mode!r}")

        # Required so gradient checkpointing works with frozen base + PEFT.
        # Not needed (and wasteful) when the encoders are fully frozen.
        if cfg.finetune_mode != "frozen":
            if hasattr(self.drug_tower, "enable_input_require_grads"):
                self.drug_tower.enable_input_require_grads()
            if hasattr(self.target_tower, "enable_input_require_grads"):
                self.target_tower.enable_input_require_grads()

        drug_hidden_size = self.drug_tower.config.hidden_size
        target_hidden_size = self.target_tower.config.hidden_size

        self.drug_proj = nn.Linear(drug_hidden_size, cfg.hidden_dim)
        self.target_proj = nn.Linear(target_hidden_size, cfg.hidden_dim)

        self.compression_factor = int(getattr(cfg, "compression_factor", 4))
        self.pooling = getattr(cfg, "pooling", "masked")
        self.fusion = getattr(cfg, "fusion", "cross_attention")
        if self.fusion not in ("cross_attention", "concat"):
            raise ValueError(f"Unknown fusion mode: {self.fusion!r}")
        self.target_compressor = nn.Conv1d(
            in_channels=cfg.hidden_dim,
            out_channels=cfg.hidden_dim,
            kernel_size=self.compression_factor,
            stride=self.compression_factor,
        )

        if self.fusion == "cross_attention":
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=cfg.hidden_dim,
                num_heads=cfg.num_attention_heads,
                dropout=cfg.dropout_rate,
                batch_first=True,
            )
            head_in = cfg.hidden_dim
        else:
            # Simple fusion baseline: concatenate pooled drug + pooled target.
            head_in = 2 * cfg.hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(head_in, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def set_encoders_trainable(self, trainable: bool):
        """Enable/disable gradients on both encoders (used for gradual unfreeze)."""
        for tower in (self.drug_tower, self.target_tower):
            for p in tower.parameters():
                p.requires_grad = trainable

    @staticmethod
    def compress_mask(attention_mask: torch.Tensor, factor: int) -> torch.Tensor:
        """Max-pool a 0/1 attention mask by ``factor`` (mirrors Conv1d stride)."""
        if factor <= 1:
            return attention_mask
        mask = attention_mask.float().unsqueeze(1)
        pooled = nn.functional.max_pool1d(mask, kernel_size=factor, stride=factor)
        return pooled.squeeze(1).long()

    def _run_drug_tower(self, input_ids, attention_mask):
        """Run MoLFormer, optionally in fp32 to avoid fp16 overflow."""
        if self.drug_tower_fp32 and input_ids.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                return self.drug_tower(input_ids=input_ids, attention_mask=attention_mask)
        return self.drug_tower(input_ids=input_ids, attention_mask=attention_mask)

    @staticmethod
    def _masked_mean(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean over positions where ``mask`` is 1 (mask shape [B, L])."""
        m = mask.unsqueeze(-1).to(states.dtype)
        denom = m.sum(dim=1).clamp(min=1.0)
        return (states * m).sum(dim=1) / denom

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, drug_input_ids, drug_attention_mask, target_input_ids, target_attention_mask):
        drug_outputs = self._run_drug_tower(drug_input_ids, drug_attention_mask)
        target_outputs = self.target_tower(
            input_ids=target_input_ids, attention_mask=target_attention_mask
        )

        drug_states = drug_outputs.last_hidden_state
        target_states = target_outputs.last_hidden_state

        drug_states = self.drug_proj(drug_states)
        target_states = self.target_proj(target_states)

        # Conv1d expects [B, C, L].
        target_states = self.target_compressor(target_states.transpose(1, 2)).transpose(1, 2)

        # Compress the protein mask to match the downsampled sequence.
        compressed_mask = self.compress_mask(target_attention_mask, self.compression_factor)
        # Guard against length mismatch caused by odd sequence lengths.
        L = target_states.size(1)
        compressed_mask = compressed_mask[:, :L]

        if self.fusion == "concat":
            drug_pooled = self._masked_mean(drug_states, drug_attention_mask)
            target_pooled = self._masked_mean(target_states, compressed_mask)
            pooled_output = torch.cat([drug_pooled, target_pooled], dim=-1)
            return self.mlp(pooled_output).squeeze(-1)

        # Cross-attention: protein (query) attends to drug (key/value).
        # Invert HF mask (0 -> True) for PyTorch's key_padding_mask.
        key_padding_mask = drug_attention_mask == 0
        attn_output, _ = self.cross_attention(
            query=target_states,
            key=drug_states,
            value=drug_states,
            key_padding_mask=key_padding_mask,
        )

        if self.pooling == "masked":
            # Masked mean over real (non-padded) protein positions.
            pooled_output = self._masked_mean(attn_output, compressed_mask)
        else:
            # Original behaviour: mean over all positions including padding.
            pooled_output = torch.mean(attn_output, dim=1)

        predictions = self.mlp(pooled_output).squeeze(-1)
        return predictions
