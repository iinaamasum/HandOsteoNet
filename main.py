#!/usr/bin/env python3
"""
HandOsteoNet: Bone Age Assessment with Attention Mechanisms
Copyright (c) 2025 Qatar University
Lead by Dr. Amith Khandakar

Main training script for HandOsteoNet model.
"""

import torch
import os
import argparse
from Dataset.dataset import create_data_loaders
from Model.model import create_model
from Training.train_val import train_model
from Evaluation.evaluation import evaluate_model
from XAI.save_grad_images import generate_gradcam_images
from Utils.utils import set_seed, create_directories


def main():
    parser = argparse.ArgumentParser(description="HandOsteoNet Training")
    parser.add_argument(
        "--train_csv", type=str, required=True, help="Path to training CSV file"
    )
    parser.add_argument(
        "--val_csv", type=str, required=True, help="Path to validation CSV file"
    )
    parser.add_argument(
        "--test_csv", type=str, required=True, help="Path to test CSV file"
    )
    parser.add_argument(
        "--train_img_dir",
        type=str,
        required=True,
        help="Path to training images directory",
    )
    parser.add_argument(
        "--val_img_dir",
        type=str,
        required=True,
        help="Path to validation images directory",
    )
    parser.add_argument(
        "--test_img_dir", type=str, required=True, help="Path to test images directory"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--num_epochs", type=int, default=150, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--warmup_epochs", type=int, default=5, help="Number of warmup epochs"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of data loader workers"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="best_bonenet.pth",
        help="Path to save best model",
    )
    parser.add_argument(
        "--generate_gradcam",
        action="store_true",
        help="Generate GradCAM images after training",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    create_directories(["GradCAM", "plots"])

    print("Creating data loaders...")
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = (
        create_data_loaders(
            train_csv_path=args.train_csv,
            val_csv_path=args.val_csv,
            test_csv_path=args.test_csv,
            train_img_dir=args.train_img_dir,
            val_img_dir=args.val_img_dir,
            test_img_dir=args.test_img_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    )

    print("Creating model...")
    model = create_model(device)

    print("Starting training...")
    metrics_data = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        warmup_epochs=args.warmup_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        model_save_path=args.model_save_path,
    )

    print("Evaluating model...")
    metrics_df, results_df = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        model_path_eval=args.model_save_path,
    )

    if args.generate_gradcam:
        print("Generating GradCAM images...")
        generate_gradcam_images(model=model, test_loader=test_loader, device=device)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
