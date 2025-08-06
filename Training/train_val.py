import torch
import pandas as pd
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from .loss_function import CombinedLoss
from .optimizer_scheduler import create_optimizer, create_scheduler


def train_epoch(model, train_loader, optimizer, criterion, scaler, device):
    """
    Train for one epoch

    Args:
        model: The model to train
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        scaler: Gradient scaler for mixed precision
        device: Device to train on

    Returns:
        train_loss: Average training loss
        train_mae: Average training MAE
    """
    model.train()
    train_loss = 0.0
    train_mae = 0.0

    for images, boneages, males, _ in tqdm(train_loader, desc="Training"):
        images, boneages, males = (
            images.to(device),
            boneages.to(device),
            males.to(device),
        )
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            outputs = model(images, males).squeeze(-1)
            loss = criterion(outputs, boneages)
            mae = torch.mean(torch.abs(outputs - boneages))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * images.size(0)
        train_mae += mae.item() * images.size(0)

    return train_loss, train_mae


def validate_epoch(model, val_loader, criterion, device):
    """
    Validate for one epoch

    Args:
        model: The model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        val_loss: Average validation loss
        val_mae: Average validation MAE
    """
    model.eval()
    val_loss = 0.0
    val_mae = 0.0

    with torch.no_grad():
        for images, boneages, males, _ in val_loader:
            images, boneages, males = (
                images.to(device),
                boneages.to(device),
                males.to(device),
            )
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                outputs = model(images, males).squeeze(-1)
                loss = criterion(outputs, boneages)
                mae = torch.mean(torch.abs(outputs - boneages))
            val_loss += loss.item() * images.size(0)
            val_mae += mae.item() * images.size(0)

    return val_loss, val_mae


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=150,
    warmup_epochs=5,
    lr=3e-4,
    weight_decay=1e-4,
    model_save_path="best_bonenet.pth",
):
    """
    Train the HandOsteoNet model

    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        warmup_epochs: Number of warmup epochs
        lr: Learning rate
        weight_decay: Weight decay
        model_save_path: Path to save the best model

    Returns:
        metrics_data: Training metrics history
    """
    device = next(model.parameters()).device
    scaler = GradScaler()
    optimizer = create_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = create_scheduler(optimizer, num_epochs, warmup_epochs)
    criterion = CombinedLoss(alpha=0.8, beta=0.1, gamma=0.1)

    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    best_val_loss = float("inf")
    metrics_data = []

    for epoch in range(num_epochs):
        train_loss, train_mae = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )
        val_loss, val_mae = validate_epoch(model, val_loader, criterion, device)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        train_mae /= len(train_loader.dataset)
        val_mae /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)

        metrics_data.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_mae": train_mae,
                "val_mae": val_mae,
            }
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model at epoch {epoch+1} with val loss: {val_loss:.4f}")

        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.8f}"
        )

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv("training_metrics.csv", index=False)

    return metrics_data
