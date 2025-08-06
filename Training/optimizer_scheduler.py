import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR


def create_optimizer(model, lr=3e-4, weight_decay=1e-4):
    """
    Create AdamW optimizer for the model

    Args:
        model: The model to optimize
        lr: Learning rate
        weight_decay: Weight decay parameter

    Returns:
        optimizer: AdamW optimizer
    """
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer, num_epochs, warmup_epochs=5):
    """
    Create learning rate scheduler with warmup and cosine annealing

    Args:
        optimizer: The optimizer
        num_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs

    Returns:
        scheduler: Combined scheduler with warmup and cosine annealing
    """
    cosine_annealing_T_max = max(1, num_epochs - warmup_epochs)
    warmup_scheduler = LambdaLR(
        optimizer, lr_lambda=lambda epoch: min(epoch / warmup_epochs, 1.0)
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=cosine_annealing_T_max, eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    return scheduler
