import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def apply_clahe_preprocessing(gray_image, clip_limit_4x4=2.0, clip_limit_8x8=4.0):
    """
    Apply CLAHE preprocessing to grayscale image

    Args:
        gray_image: Input grayscale image
        clip_limit_4x4: CLAHE clip limit for 4x4 grid
        clip_limit_8x8: CLAHE clip limit for 8x8 grid

    Returns:
        Preprocessed image with 3 channels (original, CLAHE 4x4, CLAHE 8x8)
    """
    gray = cv2.resize(gray_image, (480, 480), interpolation=cv2.INTER_LANCZOS4)
    clahe_4x4 = cv2.createCLAHE(clipLimit=clip_limit_4x4, tileGridSize=(4, 4)).apply(
        gray
    )
    clahe_8x8 = cv2.createCLAHE(clipLimit=clip_limit_8x8, tileGridSize=(8, 8)).apply(
        gray
    )
    image = np.stack([gray, clahe_4x4, clahe_8x8], axis=-1)
    return image


def get_train_transforms():
    """Get training data transforms"""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    return transforms.Compose(
        [
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(
                480, scale=(1.0, 1.2), interpolation=Image.LANCZOS
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def get_val_test_transforms():
    """Get validation/test data transforms"""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
