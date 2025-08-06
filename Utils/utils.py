import torch
import numpy as np
import random
import os


def set_seed(seed=42):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True


def create_directories(directories):
    """
    Create directories if they don't exist

    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def count_parameters(model):
    """
    Count total and trainable parameters in a model

    Args:
        model: PyTorch model

    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
