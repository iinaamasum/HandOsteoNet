#!/usr/bin/env python3
"""
HandOsteoNet Example Usage
Copyright (c) 2025 Qatar University
Lead by Dr. Amith Khandakar

This script demonstrates how to use the HandOsteoNet framework.
"""

import torch
import os
from Dataset.dataset import create_data_loaders
from Model.model import create_model
from Training.train_val import train_model
from Evaluation.evaluation import evaluate_model
from XAI.save_grad_images import generate_gradcam_images
from Utils.utils import set_seed, create_directories
from Utils.plotting import generate_regression_plots


def example_usage():
    """
    Example usage of HandOsteoNet framework
    """
    print("HandOsteoNet: Bone Age Assessment with Attention Mechanisms")
    print("Copyright (c) 2025 Qatar University")
    print("Lead by Dr. Amith Khandakar\n")

    # Set random seed for reproducibility
    set_seed(42)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories
    create_directories(["GradCAM", "plots", "models"])

    # Example data paths (replace with your actual paths)
    train_csv = "path/to/your/train.csv"
    val_csv = "path/to/your/val.csv"
    test_csv = "path/to/your/test.csv"
    train_img_dir = "path/to/your/train/images"
    val_img_dir = "path/to/your/val/images"
    test_img_dir = "path/to/your/test/images"

    # Check if data files exist
    if not all(os.path.exists(f) for f in [train_csv, val_csv, test_csv]):
        print("Warning: Data files not found. Please update the paths in this script.")
        print("Example data paths:")
        print(f"  Train CSV: {train_csv}")
        print(f"  Val CSV: {val_csv}")
        print(f"  Test CSV: {test_csv}")
        print(f"  Train images: {train_img_dir}")
        print(f"  Val images: {val_img_dir}")
        print(f"  Test images: {test_img_dir}")
        return

    print("Creating data loaders...")
    try:
        (
            train_loader,
            val_loader,
            test_loader,
            train_dataset,
            val_dataset,
            test_dataset,
        ) = create_data_loaders(
            train_csv_path=train_csv,
            val_csv_path=val_csv,
            test_csv_path=test_csv,
            train_img_dir=train_img_dir,
            val_img_dir=val_img_dir,
            test_img_dir=test_img_dir,
            batch_size=16,
            num_workers=0,
        )
        print("✓ Data loaders created successfully")
    except Exception as e:
        print(f"✗ Error creating data loaders: {e}")
        return

    print("Creating model...")
    try:
        model = create_model(device)
        print("✓ Model created successfully")
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return

    print("Starting training...")
    try:
        metrics_data = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=150,
            warmup_epochs=5,
            lr=3e-4,
            weight_decay=1e-4,
            model_save_path="models/best_bonenet.pth",
        )
        print("✓ Training completed successfully")
    except Exception as e:
        print(f"✗ Error during training: {e}")
        return

    print("Evaluating model...")
    try:
        metrics_df, results_df = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            model_path_eval="models/best_bonenet.pth",
        )
        print("✓ Evaluation completed successfully")
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        return

    print("Generating GradCAM images...")
    try:
        generate_gradcam_images(
            model=model, test_loader=test_loader, device=device, output_dir="GradCAM"
        )
        print("✓ GradCAM images generated successfully")
    except Exception as e:
        print(f"✗ Error generating GradCAM images: {e}")

    print("Generating evaluation plots...")
    try:
        generate_regression_plots(
            predictions_csv="predictions.csv",
            metrics_csv="test_metrics.csv",
            history_csv="training_metrics.csv",
            output_dir="plots",
        )
        print("✓ Evaluation plots generated successfully")
    except Exception as e:
        print(f"✗ Error generating plots: {e}")

    print("\n" + "=" * 50)
    print("HandOsteoNet Example Completed!")
    print("=" * 50)
    print("\nGenerated files:")
    print("  - models/best_bonenet.pth (best model weights)")
    print("  - training_metrics.csv (training history)")
    print("  - test_metrics.csv (evaluation metrics)")
    print("  - predictions.csv (model predictions)")
    print("  - GradCAM/ (GradCAM visualization images)")
    print("  - plots/ (evaluation plots)")
    print("\nTo run the full training with command line arguments:")
    print(
        "python main.py --train_csv path/to/train.csv --val_csv path/to/val.csv --test_csv path/to/test.csv \\"
    )
    print(
        "    --train_img_dir path/to/train/images --val_img_dir path/to/val/images --test_img_dir path/to/test/images \\"
    )
    print("    --batch_size 16 --num_epochs 150 --generate_gradcam")


if __name__ == "__main__":
    example_usage()
