import torch
from torchvision import transforms
import numpy as np


def get_inverse_normalize():
    """
    Get inverse normalization transform for ImageNet normalization

    Returns:
        Inverse normalization transform
    """
    return transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )


def inverse_preprocess_image(img_tensor):
    """
    Inverse preprocess image tensor to original scale

    Args:
        img_tensor: Normalized image tensor [C, H, W]

    Returns:
        Original scale image tensor
    """
    inv_normalize = get_inverse_normalize()
    return inv_normalize(img_tensor)


def tensor_to_numpy(img_tensor):
    """
    Convert image tensor to numpy array

    Args:
        img_tensor: Image tensor [C, H, W]

    Returns:
        Numpy array [H, W, C]
    """
    img_np = img_tensor.detach().permute(1, 2, 0).cpu().numpy()
    return np.clip(img_np, 0, 1)
