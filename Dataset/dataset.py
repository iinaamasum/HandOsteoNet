import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from ..Preprocessing.preprocessing import apply_clahe_preprocessing


class BoneAgeDataset(Dataset):
    def __init__(self, data_frame, root_dir, transform=None):
        self.data_frame = data_frame.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

        self.images = []
        self.boneages = []
        self.males = []
        self.ids = []

        for idx in tqdm(range(len(self.data_frame)), desc=f"Preloading images"):
            img_id = self.data_frame.iloc[idx]["id"]
            img_filename = f"{int(img_id)}.png"
            img_path = os.path.join(self.root_dir, img_filename)

            boneage = self.data_frame.iloc[idx]["boneage"]
            male_flag = self.data_frame.iloc[idx]["male"]

            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"[WARNING] Missing image: {img_path}")
                continue

            image = apply_clahe_preprocessing(gray)

            self.images.append(image)
            self.boneages.append(boneage)
            self.males.append(1 if male_flag else 0)
            self.ids.append(img_id)

        self.images = np.stack(self.images, axis=0)
        self.boneages = torch.tensor(self.boneages, dtype=torch.float32)
        self.males = torch.tensor(self.males, dtype=torch.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        boneage = self.boneages[idx]
        male = self.males[idx]
        img_id = self.ids[idx]
        return image, boneage, male, img_id


def create_data_loaders(
    train_csv_path,
    val_csv_path,
    test_csv_path,
    train_img_dir,
    val_img_dir,
    test_img_dir,
    batch_size=16,
    num_workers=0,
):
    """
    Create data loaders for training, validation, and test sets

    Args:
        train_csv_path: Path to training CSV file
        val_csv_path: Path to validation CSV file
        test_csv_path: Path to test CSV file
        train_img_dir: Path to training images directory
        val_img_dir: Path to validation images directory
        test_img_dir: Path to test images directory
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    """
    from ..Preprocessing.preprocessing import (
        get_train_transforms,
        get_val_test_transforms,
    )

    train_transform = get_train_transforms()
    val_test_transform = get_val_test_transforms()

    train_df = pd.read_csv(train_csv_path)
    train_df = train_df.reset_index(drop=True)
    val_df = pd.read_csv(val_csv_path)
    val_df = val_df.reset_index(drop=True)
    test_df = pd.read_csv(test_csv_path)
    test_df = test_df.reset_index(drop=True)

    train_dataset = BoneAgeDataset(
        data_frame=train_df, root_dir=train_img_dir, transform=train_transform
    )
    val_dataset = BoneAgeDataset(
        data_frame=val_df, root_dir=val_img_dir, transform=val_test_transform
    )
    test_dataset = BoneAgeDataset(
        data_frame=test_df, root_dir=test_img_dir, transform=val_test_transform
    )

    print(f"Number of train samples: {len(train_dataset)}")
    print(f"Number of val   samples: {len(val_dataset)}")
    print(f"Number of test  samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    )
