import torch
from dataclasses import dataclass

@dataclass
class Config:
    # ---------------------------------------------------------
    # 1. Global Settings
    # ---------------------------------------------------------
    project_name: str = "cortex-dti"
    seed: int = 42
    
    # Auto-detect if your bare-metal setup has CUDA available
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------
    # 2. Data & Paths
    # ---------------------------------------------------------
    train_path: str = "data/processed/train.parquet"
    val_path: str = "data/processed/val.parquet"
    test_path: str = "data/processed/test.parquet"
    
    # The safety limits we discovered during EDA
    max_drug_len: int = 200
    max_target_len: int = 1000

    # ---------------------------------------------------------
    # 3. Model Architecture (Hugging Face)
    # ---------------------------------------------------------
    # Tower 1: The Chemical Reader
    drug_model_name: str = "ibm/MoLFormer-XL-both-10pct"
    
    # Tower 2: The Protein Reader (Using the highly efficient 8M parameter version)
    target_model_name: str = "facebook/esm2_t6_8M_UR50D"
    
    # The Bridge: Cross-Attention
    hidden_dim: int = 256
    num_attention_heads: int = 8       # UPGRADED: From 4 to 8 for better "searching"
    dropout_rate: float = 0.1          # UPGRADED: To protect the new 3-layer MLP from overfitting

    # ---------------------------------------------------------
    # 4. LoRA (Low-Rank Adaptation) Settings
    # ---------------------------------------------------------
    lora_r: int = 16                   # UPGRADED: Doubled capacity for complex chemistry
    lora_alpha: int = 32               # UPGRADED: Scaled to match the new rank
    lora_dropout: float = 0.1
    # We apply LoRA to the attention projection matrices AND the dense layers now
    target_modules: tuple = ("query", "key", "value", "dense")

    # ---------------------------------------------------------
    # 5. Training Hyperparameters
    # ---------------------------------------------------------
    batch_size: int = 32               # UPGRADED: Bumped from 16 to leverage Kaggle T4 GPUs
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 0.05         # UPGRADED: Heavier penalty for overfitting
    early_stopping_patience: int = 3
    
    # ---------------------------------------------------------
    # 6. Online Hard Example Mining (OHEM)
    # ---------------------------------------------------------
    # Wait until Epoch 5 to start mining so the model establishes a baseline first
    hard_mining_start_epoch: int = 5
    # When mining is active, only calculate gradients on the hardest 50% of the batch
    hard_mining_fraction: float = 0.5