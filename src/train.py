import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Append root directory to path so we can import config natively
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config
from src.logger import get_logger
from src.tokenization import BindingDataset
from src.models import BindingModel

logger = get_logger(__name__)

# ---------------------------------------------------------
# THE UPGRADE: DYNAMIC PADDING COLLATOR
# Prevents wasting GPU FLOPs on empty padding tokens
# ---------------------------------------------------------
def dynamic_collate_fn(batch):
    """Trims pre-padded sequences down to the absolute max length needed for this specific batch."""
    # Find the real max lengths in this batch by counting the 1s in the attention masks
    max_drug = max([(item["drug_attention_mask"] == 1).sum().item() for item in batch])
    max_target = max([(item["target_attention_mask"] == 1).sum().item() for item in batch])
    
    # Slice the tensors to strip away the wasted empty space
    for item in batch:
        item["drug_input_ids"] = item["drug_input_ids"][:max_drug]
        item["drug_attention_mask"] = item["drug_attention_mask"][:max_drug]
        item["target_input_ids"] = item["target_input_ids"][:max_target]
        item["target_attention_mask"] = item["target_attention_mask"][:max_target]
        
    return torch.utils.data.dataloader.default_collate(batch)

def train_one_epoch(model, dataloader, optimizer, criterion_train, device, epoch):
    """Runs a single training pass with dynamic Hard Negative Mining."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        drug_ids = batch["drug_input_ids"].to(device)
        drug_mask = batch["drug_attention_mask"].to(device)
        target_ids = batch["target_input_ids"].to(device)
        target_mask = batch["target_attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        predictions = model(
            drug_input_ids=drug_ids,
            drug_attention_mask=drug_mask,
            target_input_ids=target_ids,
            target_attention_mask=target_mask
        )
        
        # Calculate loss per sample (reduction='none' allows us to see individual scores)
        sample_losses = criterion_train(predictions, labels)
        
        # ---------------------------------------------------------
        # HARD NEGATIVE MINING (Dynamic Batching)
        # After a warmup period, only backpropagate on the hardest examples
        # ---------------------------------------------------------
        if epoch >= Config.hard_mining_start_epoch:
            # Keep only the fraction of the batch with the highest error
            num_hard = max(1, int(sample_losses.size(0) * Config.hard_mining_fraction))
            hard_losses, _ = torch.topk(sample_losses, k=num_hard)
            loss = hard_losses.mean()
        else:
            # Standard training: average the loss across all samples
            loss = sample_losses.mean()
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 50 == 0:
            status = " [HARD MINING ACTIVE]" if epoch >= Config.hard_mining_start_epoch else ""
            logger.info(f"  Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}{status}")
            
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion_val, device):
    """Evaluates the model on unseen validation data."""
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
            
            loss = criterion_val(predictions, labels)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def main():
    logger.info(f"Starting Training Process for {Config.project_name}")
    logger.info(f"Target Execution Device Detected: {Config.device}")
    
    checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

    logger.info("Loading Train and Validation Datasets...")
    train_dataset = BindingDataset(data_path=Config.train_path)
    val_dataset = BindingDataset(data_path=Config.val_path)
    
    # ---------------------------------------------------------
    # INJECTED THE CUSTOM COLLATOR HERE
    # ---------------------------------------------------------
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.batch_size, 
        shuffle=True, 
        drop_last=True,
        collate_fn=dynamic_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.batch_size, 
        shuffle=False,
        collate_fn=dynamic_collate_fn
    )
    
    model = BindingModel().to(Config.device)
    
    # OPTIMIZATION SUITE UPGRADES
    criterion_train = nn.MSELoss(reduction='none')
    criterion_val = nn.MSELoss()
    
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=1e-6)
    
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(1, Config.epochs + 1):
        logger.info(f"--- Epoch {epoch}/{Config.epochs} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_train, Config.device, epoch)
        logger.info(f"Epoch {epoch} Complete | Avg Training Loss: {train_loss:.4f}")
        
        logger.info("Running validation check...")
        val_loss = validate(model, val_loader, criterion_val, Config.device)
        logger.info(f"Epoch {epoch} Complete | Avg Validation Loss: {val_loss:.4f}")
        
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"Learning Rate mapped to: {current_lr:.6f} for next epoch.")
        scheduler.step()
        
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