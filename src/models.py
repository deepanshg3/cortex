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
        # We use deterministic_eval to ensure reproducible layers
        self.drug_tower = AutoModel.from_pretrained(Config.drug_model_name, deterministic_eval=True, trust_remote_code=True)
        
        logger.info("Building Tower 2: Protein Model (ESM-2)")
        self.target_tower = AutoModel.from_pretrained(Config.target_model_name)
        
        # ---------------------------------------------------------
        # APPLY LoRA (Low-Rank Adaptation)
        # Freezes the base models and injects lightweight trainable layers
        # ---------------------------------------------------------
        lora_config = LoraConfig(
            r=Config.lora_r,
            lora_alpha=Config.lora_alpha,
            target_modules=list(Config.target_modules),
            lora_dropout=Config.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        
        logger.info("Injecting LoRA adapters into attention layers...")
        self.drug_tower = get_peft_model(self.drug_tower, lora_config)
        self.target_tower = get_peft_model(self.target_tower, lora_config)
        
        # ---------------------------------------------------------
        # DIMENSION ALIGNMENT PROJECTIONS
        # MolFormer outputs 768 dims. ESM-2 8M outputs 320 dims.
        # We project them to the SAME dimension (256) for Cross-Attention.
        # ---------------------------------------------------------
        drug_hidden_size = self.drug_tower.config.hidden_size # 768
        target_hidden_size = self.target_tower.config.hidden_size # 320
        
        self.drug_proj = nn.Linear(drug_hidden_size, Config.hidden_dim)
        self.target_proj = nn.Linear(target_hidden_size, Config.hidden_dim)
        
        # ---------------------------------------------------------
        # THE BRIDGE: CROSS-ATTENTION
        # ---------------------------------------------------------
        # batch_first=True means tensors are (Batch, Seq_Len, Dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=Config.hidden_dim, 
            num_heads=Config.num_attention_heads, 
            dropout=Config.dropout_rate,
            batch_first=True
        )
        
        # ---------------------------------------------------------
        # THE OUTPUT HEAD (Regression)
        # Compresses the final pooled representation into a single Kd value
        # ---------------------------------------------------------
        self.mlp = nn.Sequential(
            nn.Linear(Config.hidden_dim, Config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(Config.dropout_rate),
            nn.Linear(Config.hidden_dim // 2, 1) # Predicts 1 continuous value
        )

    def forward(self, drug_input_ids, drug_attention_mask, target_input_ids, target_attention_mask):
        """
        The Forward Pass: How data flows through the network graph.
        """
        # 1. Pass strings through the base towers
        drug_outputs = self.drug_tower(input_ids=drug_input_ids, attention_mask=drug_attention_mask)
        target_outputs = self.target_tower(input_ids=target_input_ids, attention_mask=target_attention_mask)
        
        drug_states = drug_outputs.last_hidden_state
        target_states = target_outputs.last_hidden_state
        
        # 2. Project down to shared mathematical space
        # shape becomes: (Batch, Seq_Len, Config.hidden_dim)
        drug_states = self.drug_proj(drug_states)
        target_states = self.target_proj(target_states)
        
        # 3. Cross-Attention
        # Protein (Target) acts as the Query looking at the Drug (Key/Value)
        # PyTorch attention needs key_padding_mask where padding tokens are True (1)
        # HF masks use 0 for padding. So we invert it with == 0.
        key_padding_mask = (drug_attention_mask == 0)
        
        attn_output, _ = self.cross_attention(
            query=target_states,
            key=drug_states,
            value=drug_states,
            key_padding_mask=key_padding_mask
        )
        
        # 4. Mean Pooling
        # Compress the sequence length down into a single dense vector per batch
        pooled_output = torch.mean(attn_output, dim=1)
        
        # 5. Predict Binding Affinity (Kd)
        predictions = self.mlp(pooled_output).squeeze(-1)
        
        return predictions

# ---------------------------------------------------------------------
# Debugging / Sanity Check Loop
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Executing Neural Network Graph Verification")
    try:
        # Instantiate the entire architecture
        model = BindingModel()
        logger.info("Model successfully constructed and LoRA adapters injected.")
        
        # Print trainable parameter percentage
        model.drug_tower.print_trainable_parameters()
        
        # Create dummy tensors matching our exact Config specs (Batch Size of 2)
        batch_size = 2
        
        # Fetch actual vocabulary boundaries dynamically to prevent out-of-bounds indices
        drug_vocab = model.drug_tower.config.vocab_size
        target_vocab = model.target_tower.config.vocab_size
        
        dummy_drug_ids = torch.randint(0, drug_vocab, (batch_size, Config.max_drug_len))
        dummy_drug_mask = torch.ones(batch_size, Config.max_drug_len, dtype=torch.long)
        
        dummy_target_ids = torch.randint(0, target_vocab, (batch_size, Config.max_target_len))
        dummy_target_mask = torch.ones(batch_size, Config.max_target_len, dtype=torch.long)
        
        logger.info("Pushing dummy batch through the graph...")
        
        # Disable gradient calculation for testing to save memory
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