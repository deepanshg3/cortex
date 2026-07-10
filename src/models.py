import os
import sys
import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

# Append root directory to path so we can import config natively
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config
from src.logger import get_logger

logger = get_logger(__name__)

class BindingModel(nn.Module):
    def __init__(self):
        super(BindingModel, self).__init__()
        
        logger.info("Building Tower 1: Chemical Model (MolFormer)")
        self.drug_tower = AutoModel.from_pretrained(Config.drug_model_name, deterministic_eval=True, trust_remote_code=True)
        
        logger.info("Building Tower 2: Protein Model (ESM-2)")
        self.target_tower = AutoModel.from_pretrained(Config.target_model_name)
        
        # --- FIX: Enable Gradient Checkpointing to save VRAM ---
        if hasattr(self.drug_tower, "gradient_checkpointing_enable"):
            self.drug_tower.gradient_checkpointing_enable()
        if hasattr(self.target_tower, "gradient_checkpointing_enable"):
            self.target_tower.gradient_checkpointing_enable()

        # --- FIX: Separate configurations to avoid cross-mutation warning ---
        drug_lora_config = LoraConfig(
            r=Config.lora_r,
            lora_alpha=Config.lora_alpha,
            target_modules=list(Config.target_modules),
            lora_dropout=Config.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        
        target_lora_config = LoraConfig(
            r=Config.lora_r,
            lora_alpha=Config.lora_alpha,
            target_modules=list(Config.target_modules),
            lora_dropout=Config.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        
        logger.info("Injecting High-Capacity LoRA adapters...")
        self.drug_tower = get_peft_model(self.drug_tower, drug_lora_config)
        self.target_tower = get_peft_model(self.target_tower, target_lora_config)

        # --- CRITICAL SAFETY FIX FOR GRADIENT CHECKPOINTING ---
        # Ensures PyTorch doesn't crash during backward pass with frozen base models
        if hasattr(self.drug_tower, "enable_input_require_grads"):
            self.drug_tower.enable_input_require_grads()
        if hasattr(self.target_tower, "enable_input_require_grads"):
            self.target_tower.enable_input_require_grads()
        
        # ---------------------------------------------------------
        # DIMENSION ALIGNMENT PROJECTIONS
        # ---------------------------------------------------------
        drug_hidden_size = self.drug_tower.config.hidden_size # Usually 768
        target_hidden_size = self.target_tower.config.hidden_size # Usually 320
        
        self.drug_proj = nn.Linear(drug_hidden_size, Config.hidden_dim)
        self.target_proj = nn.Linear(target_hidden_size, Config.hidden_dim)
        
        # ---------------------------------------------------------
        # THE UPGRADE: SPATIAL REDUCTION COMPRESSOR
        # Prevents O(N^2) memory blowout on edge hardware
        # ---------------------------------------------------------
        self.target_compressor = nn.Conv1d(
            in_channels=Config.hidden_dim, 
            out_channels=Config.hidden_dim, 
            kernel_size=4, 
            stride=4
        )
        
        # ---------------------------------------------------------
        # THE BRIDGE: MULTI-HEAD CROSS-ATTENTION
        # ---------------------------------------------------------
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=Config.hidden_dim, 
            num_heads=Config.num_attention_heads, 
            dropout=Config.dropout_rate,
            batch_first=True
        )
        
        # ---------------------------------------------------------
        # THE UPGRADED OUTPUT HEAD (3-Layer GELU MLP)
        # Gives the network more non-linear reasoning power
        # ---------------------------------------------------------
        self.mlp = nn.Sequential(
            nn.Linear(Config.hidden_dim, Config.hidden_dim),
            nn.GELU(),
            nn.Dropout(Config.dropout_rate),
            nn.Linear(Config.hidden_dim, Config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(Config.dropout_rate),
            nn.Linear(Config.hidden_dim // 2, 1)
        )

    def forward(self, drug_input_ids, drug_attention_mask, target_input_ids, target_attention_mask):
        """
        The Forward Pass: Mathematical Molecular Docking
        """
        # 1. Base Towers
        drug_outputs = self.drug_tower(input_ids=drug_input_ids, attention_mask=drug_attention_mask)
        target_outputs = self.target_tower(input_ids=target_input_ids, attention_mask=target_attention_mask)
        
        # Pull 3D tensors, not the pooled 1D summaries
        drug_states = drug_outputs.last_hidden_state
        target_states = target_outputs.last_hidden_state
        
        # 2. Project to shared mathematical space
        drug_states = self.drug_proj(drug_states)
        target_states = self.target_proj(target_states)
        
        # ---------------------------------------------------------
        # COMPRESSION EXECUTION
        # ---------------------------------------------------------
        # PyTorch Conv1D expects [Batch, Channels, Seq_Len]
        target_states = target_states.transpose(1, 2)
        target_states = self.target_compressor(target_states)
        target_states = target_states.transpose(1, 2)
        
        # Compress the attention mask to match the new shorter sequence
        mask_compressor = nn.MaxPool1d(kernel_size=4, stride=4)
        float_mask = target_attention_mask.float().unsqueeze(1)
        compressed_mask = mask_compressor(float_mask).squeeze(1).long()
        
        # 3. Cross-Attention (Protein searches the Drug)
        # Invert HF masks (0 -> True) for PyTorch's key_padding_mask
        key_padding_mask = (drug_attention_mask == 0)
        
        attn_output, _ = self.cross_attention(
            query=target_states,
            key=drug_states,
            value=drug_states,
            key_padding_mask=key_padding_mask
        )
        
        # 4. Mean Pooling (Compress sequence to a single vector)
        pooled_output = torch.mean(attn_output, dim=1)
        
        # 5. Predict final pKd value
        predictions = self.mlp(pooled_output).squeeze(-1)
        
        return predictions

if __name__ == "__main__":
    logger.info("Executing Neural Network Graph Verification")
    try:
        model = BindingModel()
        model.drug_tower.print_trainable_parameters()
        
        batch_size = 2
        drug_vocab = model.drug_tower.config.vocab_size
        target_vocab = model.target_tower.config.vocab_size
        
        dummy_drug_ids = torch.randint(0, drug_vocab, (batch_size, Config.max_drug_len))
        dummy_drug_mask = torch.ones(batch_size, Config.max_drug_len, dtype=torch.long)
        
        dummy_target_ids = torch.randint(0, target_vocab, (batch_size, Config.max_target_len))
        dummy_target_mask = torch.ones(batch_size, Config.max_target_len, dtype=torch.long)
        
        with torch.no_grad():
            preds = model(
                drug_input_ids=dummy_drug_ids,
                drug_attention_mask=dummy_drug_mask,
                target_input_ids=dummy_target_ids,
                target_attention_mask=dummy_target_mask
            )
        
        logger.info(f"Prediction Tensor Shape: {preds.shape}")
        logger.info("Graph verification passed. Matrix multiplication aligns perfectly.")
        
    except Exception as e:
        logger.error(f"Network verification failed: {str(e)}")