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

def dynamic_collate_fn(batch):
    """Trims pre-padded sequences down to the absolute max length needed for this specific batch."""
    max_drug = max([(item["drug_attention_mask"] == 1).sum().item() for item in batch])
    max_target = max([(item["target_attention_mask"] == 1).sum().item() for item in batch])

    for item in batch:
        item["drug_input_ids"] = item["drug_input_ids"][:max_drug]
        item["drug_attention_mask"] = item["drug_attention_mask"][:max_drug]
        item["target_input_ids"] = item["target_input_ids"][:max_target]
        item["target_attention_mask"] = item["target_attention_mask"][:max_target]

    return torch.utils.data.dataloader.default_collate(batch)

# --- UPGRADE: Injected GradScaler for FP16 ---
def train_one_epoch(model, dataloader, optimizer, criterion_train, device, epoch, scaler):
    """Runs a single training pass with dynamic Hard Negative Mining and FP16 Precision."""
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        drug_ids = batch["drug_input_ids"].to(device)
        drug_mask = batch["drug_attention_mask"].to(device)
        target_ids = batch["target_input_ids"].to(device)
        target_mask = batch["target_attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        # --- UPGRADE: Native FP16 Mixed Precision Math ---
        with torch.amp.autocast('cuda'):
            predictions = model(
                drug_input_ids=drug_ids,
                drug_attention_mask=drug_mask,
                target_input_ids=target_ids,
                target_attention_mask=target_mask
            )

            sample_losses = criterion_train(predictions, labels)

            if epoch >= Config.hard_mining_start_epoch:
                num_hard = max(1, int(sample_losses.size(0) * Config.hard_mining_fraction))
                hard_losses, _ = torch.topk(sample_losses, k=num_hard)
                loss = hard_losses.mean()
            else:
                loss = sample_losses.mean()

        # --- UPGRADE: Scaled Backpropagation ---
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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

            # Use FP16 for validation to keep things fast
            with torch.amp.autocast('cuda'):
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

    # --- UPGRADE: Data Pipeline Unblocker (num_workers & pin_memory) ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=dynamic_collate_fn,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        collate_fn=dynamic_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    model = BindingModel().to(Config.device)

    criterion_train = nn.MSELoss(reduction='none')
    criterion_val = nn.MSELoss()

    # Initialize optimization suite BEFORE loading checkpoint
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0

    # --- UPGRADE: Full Checkpoint Auto-Resume ---
    if os.path.exists(checkpoint_path):
        logger.info(f"Found existing checkpoint! Rebuilding previous state from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=Config.device)

        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']

        logger.info(f"Successfully resurrected training. Resuming straight from Epoch {start_epoch}")

    # --- UPGRADE: Dual-GPU Activation ---
    if torch.cuda.device_count() > 1:
        logger.info(f"Turbo boost: Activating {torch.cuda.device_count()} GPUs via DataParallel!")
        model = nn.DataParallel(model)

    # The Loop Clock uses the dynamic start_epoch
    for epoch in range(start_epoch, Config.epochs + 1):
        logger.info(f"--- Epoch {epoch}/{Config.epochs} ---")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_train, Config.device, epoch, scaler)
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
            logger.info(f"New best validation score! Saving Full Checkpoint to: {checkpoint_path}")

            # Strip the DataParallel wrapper before saving so it loads cleanly on the bare-metal Jetson
            state_dict_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

            # --- UPGRADE: Full Dictionary Checkpoint ---
            checkpoint = {
                'epoch': epoch,
                'model_state': state_dict_to_save,
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, checkpoint_path)
        else:
            patience_counter += 1
            logger.info(f"Validation loss did not improve. Early Stopping patience: {patience_counter}/{Config.early_stopping_patience}")

        if patience_counter >= Config.early_stopping_patience:
            logger.warning(f"Early Stopping triggered at Epoch {epoch}. Training halted to prevent overfitting.")
            break

    logger.info("Training pipeline cycle finished completely.")

if __name__ == "__main__":
    main()