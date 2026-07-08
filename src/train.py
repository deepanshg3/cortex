import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Append root directory to path so we can import config natively
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config
from src.logger import get_logger
from src.tokenization import BindingDataset
from src.models import BindingModel

logger = get_logger(__name__)

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Runs a single training pass over the entire training dataset."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move all input tensors to the target device (CPU/CUDA)
        drug_ids = batch["drug_input_ids"].to(device)
        drug_mask = batch["drug_attention_mask"].to(device)
        target_ids = batch["target_input_ids"].to(device)
        target_mask = batch["target_attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(
            drug_input_ids=drug_ids,
            drug_attention_mask=drug_mask,
            target_input_ids=target_ids,
            target_attention_mask=target_mask
        )
        
        # Calculate loss (MSE)
        loss = criterion(predictions, labels)
        
        # Backward pass & weight adjustment
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log status every 50 batches so we don't spam the output terminal
        if batch_idx % 50 == 0:
            logger.info(f"  Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
            
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """Evaluates the model on unseen validation data to detect overfitting."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            drug_ids = batch["drug_input_ids"].to(device)
            drug_mask = batch["drug_attention_mask"].to(device)
            target_ids = batch["target_input_ids"].to(device)
            target_mask = batch["target_attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            predictions = model(
                drug_input_ids=drug_ids,
                drug_attention_mask=drug_mask,
                target_input_ids=target_ids,
                target_attention_mask=target_mask
            )
            
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def main():
    logger.info(f"Starting Training Process for {Config.project_name}")
    logger.info(f"Target Execution Device Detected: {Config.device}")
    
    # 1. Create a directory to store our best model checkpoints
    checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

    # 2. Instantiate PyTorch Datasets
    logger.info("Loading Train and Validation Datasets...")
    train_dataset = BindingDataset(data_path=Config.train_path)
    val_dataset = BindingDataset(data_path=Config.val_path)
    
    # 3. Spin up DataLoaders (Handles batching, shuffling, and workers)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.batch_size, 
        shuffle=True, 
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.batch_size, 
        shuffle=False
    )
    
    # 4. Initialize Network Architecture
    model = BindingModel().to(Config.device)
    
    # 5. Define Optimization Suite
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    
    # 6. Training Controller Variables
    best_val_loss = float("inf")
    patience_counter = 0
    
    # 7. The Core Training Loop
    for epoch in range(1, Config.epochs + 1):
        logger.info(f"--- Epoch {epoch}/{Config.epochs} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, Config.device)
        logger.info(f"Epoch {epoch} Complete | Avg Training Loss: {train_loss:.4f}")
        
        logger.info("Running validation check...")
        val_loss = validate(model, val_loader, criterion, Config.device)
        logger.info(f"Epoch {epoch} Complete | Avg Validation Loss: {val_loss:.4f}")
        
        # Early Stopping & Checkpoint Saving Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            logger.info(f"New best validation score achieved! Saving state parameters to: {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            logger.info(f"Validation loss did not improve. Early Stopping patience: {patience_counter}/{Config.early_stopping_patience}")
            
        if patience_counter >= Config.early_stopping_patience:
            logger.warning(f"Early Stopping triggered at Epoch {epoch}. Training halted to prevent overfitting.")
            break
            
    logger.info("Training pipeline cycle finished completely.")

if __name__ == "__main__":
    main()