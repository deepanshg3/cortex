import os
import sys
import torch
from torch.utils.data import Dataset
import polars as pl
from transformers import AutoTokenizer

# Append root directory to path so we can import config natively
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config
from src.logger import get_logger

# Initialize the centralized logger for this specific file
logger = get_logger(__name__)

class BindingDataset(Dataset):
    def __init__(self, data_path: str):
        """
        Custom PyTorch Dataset for Drug-Target Interaction tracking.
        Reads highly optimized Parquet files directly.
        """
        # 1. Load the Parquet data shard
        if not os.path.exists(data_path):
            logger.error(f"Data path not found at: {data_path}")
            raise FileNotFoundError(f"Data path not found at: {data_path}. Ensure data/processed steps ran successfully.")
        
        self.df = pl.read_parquet(data_path)
        
        # 2. Extract explicit numpy-backed arrays for ultra-fast indexing
        self.drugs = self.df.get_column("Drug").to_numpy()
        self.targets = self.df.get_column("Target").to_numpy()
        
        # --- FIXED: Points dynamically to our precomputed professional 'pkd' column ---
        self.labels = self.df.get_column("pkd").to_numpy()
        
        # 3. Initialize separate tokenizers from Hugging Face hub
        logger.info(f"Initializing Tokenizer for Chemical Tower ({Config.drug_model_name})...")
        self.drug_tokenizer = AutoTokenizer.from_pretrained(Config.drug_model_name, trust_remote_code=True)
        
        logger.info(f"Initializing Tokenizer for Protein Tower ({Config.target_model_name})...")
        self.target_tokenizer = AutoTokenizer.from_pretrained(Config.target_model_name)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        # Pull strings from our localized index positions
        drug_str = str(self.drugs[idx])
        target_str = str(self.targets[idx])
        label = float(self.labels[idx])
        
        # Tokenize the Chemical String (MolFormer)
        drug_tokens = self.drug_tokenizer(
            drug_str,
            max_length=Config.max_drug_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize the Protein Sequence (ESM-2)
        target_tokens = self.target_tokenizer(
            target_str,
            max_length=Config.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Return a dictionary containing clean PyTorch tensors
        return {
            # Squeeze removes the unnecessary batch dimension (1, Seq_Len) -> (Seq_Len)
            "drug_input_ids": drug_tokens["input_ids"].squeeze(0),
            "drug_attention_mask": drug_tokens["attention_mask"].squeeze(0),
            "target_input_ids": target_tokens["input_ids"].squeeze(0),
            "target_attention_mask": target_tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float32)
        }

# ---------------------------------------------------------------------
# Debugging / Sanity Check Loop
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Executing Tokenization Integration Test")
    try:
        # Load the validation dataset shard to run a quick trace test
        test_dataset = BindingDataset(data_path=Config.val_path)
        logger.info(f"Dataset successfully parsed! Total samples loaded: {len(test_dataset)}")
        
        # Pull a single tokenized batch item to verify data shape completeness
        sample = test_dataset[0]
        logger.info("--- Structural Shapes for Neural Network Processing ---")
        logger.info(f"Drug Input Tensor Shape:   {sample['drug_input_ids'].shape}")
        logger.info(f"Target Input Tensor Shape: {sample['target_input_ids'].shape}")
        logger.info(f"Ground Truth Affinity (pK_d): {sample['label'].item():.4f}")
        logger.info("Pipeline validation step complete. Ready for neural building.")
        
    except Exception as e:
        logger.error(f"Tokenizer verification step failed: {str(e)}")