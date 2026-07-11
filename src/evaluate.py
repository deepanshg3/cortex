import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# Append root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config
from src.logger import get_logger
from src.tokenization import BindingDataset
from src.models import BindingModel

logger = get_logger(__name__)

def evaluate_model():
    logger.info("--- Initiating Zero-Shot Evaluation (The Final Exam) ---")
    
    # 1. Device Selection (Will use CPU locally if no GPU is found)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Inference Engine mapped to: {device}")
    
    # 2. Load the Cold-Target Test Dataset
    logger.info("Loading Disease X Test Data...")
    test_dataset = BindingDataset(data_path=Config.test_path)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)
    
    # 3. Construct Architecture and Load Peak Weights
    logger.info("Constructing Network and loading optimal weights...")
    model = BindingModel().to(device)
    
    checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_model_2.pt"))
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at {checkpoint_path}. Did you rename the downloaded file?")
        return
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # 4. Inference Loop
    all_preds = []
    all_truths = []
    
    logger.info("Running forward pass on all test pairs. This may take a few minutes on CPU...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            drug_ids = batch["drug_input_ids"].to(device)
            drug_mask = batch["drug_attention_mask"].to(device)
            target_ids = batch["target_input_ids"].to(device)
            target_mask = batch["target_attention_mask"].to(device)
            labels = batch["label"].cpu().numpy()
            
            predictions = model(
                drug_input_ids=drug_ids,
                drug_attention_mask=drug_mask,
                target_input_ids=target_ids,
                target_attention_mask=target_mask
            ).cpu().numpy()
            
            all_preds.extend(predictions)
            all_truths.extend(labels)
            
            if batch_idx % 20 == 0:
                logger.info(f"Processed {batch_idx}/{len(test_loader)} batches...")
                
    # 5. Calculate Final Metrics
    all_preds = np.array(all_preds)
    all_truths = np.array(all_truths)
    
    mse = mean_squared_error(all_truths, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_truths, all_preds)
    pearson_corr, _ = pearsonr(all_truths, all_preds)
    
    logger.info("====================================================")
    logger.info(" FINAL EXAM RESULTS (UNSEEN TARGETS)")
    logger.info("====================================================")
    logger.info(f" Mean Squared Error (MSE) : {mse:.4f}")
    logger.info(f" Root Mean Sq Error (RMSE): {rmse:.4f} pKd")
    logger.info(f" R-Squared (R2) Score     : {r2:.4f}")
    logger.info(f" Pearson Correlation (r)  : {pearson_corr:.4f}")
    logger.info("====================================================")

if __name__ == "__main__":
    evaluate_model()