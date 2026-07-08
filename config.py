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
    num_attention_heads: int = 4
    dropout_rate: float = 0.1

    # ---------------------------------------------------------
    # 4. LoRA (Low-Rank Adaptation) Settings
    # ---------------------------------------------------------
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # We apply LoRA to the attention projection matrices
    target_modules: tuple = ("query", "key", "value")

    # ---------------------------------------------------------
    # 5. Training Hyperparameters
    # ---------------------------------------------------------
    # Kept small to fit comfortably in bare-metal GPU memory
    batch_size: int = 16
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    early_stopping_patience: int = 3