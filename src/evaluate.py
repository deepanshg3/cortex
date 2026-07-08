import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

# Append root directory to path so we can import config natively
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config
from src.logger import get_logger
from src.tokenization import BindingDataset
from src.models import BindingModel

logger = get_logger(__name__)

def evaluate_model():
    logger.info("--- Initiating Zero-Shot Model Evaluation ---")
    
    # 1. Verify Checkpoint Exists
    checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_model.pt"))
    if not os.path.exists(checkpoint_path):
        logger.error(f"No trained weights found at {checkpoint_path}. Run train.py first!")
        return

    # 2. Load the Unseen Test Data (Disease X)
    logger.info(f"Loading Test Dataset from {Config.test_path}...")
    test_dataset = BindingDataset(data_path=Config.test_path)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)
    
    # 3. Build Architecture and Load Trained Weights
    logger.info("Constructing Network Architecture...")
    model = BindingModel().to(Config.device)
    
    logger.info("Loading best fine-tuned LoRA weights...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=Config.device))
    model.eval()

    # 4. Storage for Predictions and Ground Truth
    all_preds = []
    all_labels = []

    logger.info("Running inference on test set. This may take a moment...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            drug_ids = batch["drug_input_ids"].to(Config.device)
            drug_mask = batch["drug_attention_mask"].to(Config.device)
            target_ids = batch["target_input_ids"].to(Config.device)
            target_mask = batch["target_attention_mask"].to(Config.device)
            labels = batch["label"].to(Config.device)
            
            predictions = model(
                drug_input_ids=drug_ids,
                drug_attention_mask=drug_mask,
                target_input_ids=target_ids,
                target_attention_mask=target_mask
            )
            
            # Move data back to CPU and convert to standard numpy arrays for sklearn
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if batch_idx % 50 == 0 and batch_idx > 0:
                logger.info(f"Processed {batch_idx}/{len(test_loader)} batches...")

    # 5. Calculate Mathematical Metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    pearson_corr, _ = pearsonr(all_labels, all_preds)
    spearman_corr, _ = spearmanr(all_labels, all_preds)

    # 6. Final Report
    logger.info("==================================================")
    logger.info("FINAL EVALUATION METRICS (TEST SET)")
    logger.info("==================================================")
    logger.info(f"Total Samples Evaluated: {len(all_labels)}")
    logger.info(f"MSE (Mean Squared Error):  {mse:.4f}")
    logger.info(f"RMSE:                      {rmse:.4f}")
    logger.info(f"Pearson Correlation (r):   {pearson_corr:.4f}")
    logger.info(f"Spearman Correlation (ρ):  {spearman_corr:.4f}")
    logger.info("==================================================")

if __name__ == "__main__":
    evaluate_model()