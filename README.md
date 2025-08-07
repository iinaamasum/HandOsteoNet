# HandOsteoNet - Advanced Bone Age Assessment System

**Copyright (c) 2025 Qatar University**  
**Lead by Dr. Amith Khandakar**

## Overview

HandOsteoNet is a deep learning framework for automated bone age assessment using hand radiographs. The model combines segmentation and regression components with attention mechanisms (CBAM and Self-Attention) to achieve accurate bone age predictions.

## Architecture

The model consists of two main components:

1. **Segmentation Module**: RegNet-Y-400MF encoder with U-Net style decoder
2. **Prediction Module**: Custom CNN with CBAM, Self-Attention, and Depthwise Separable Convolutions

## Key Features

-   **Multi-channel preprocessing**: CLAHE enhancement with different grid sizes
-   **Attention mechanisms**: CBAM (Convolutional Block Attention Module) and Self-Attention
-   **Mixed precision training**: Automatic mixed precision for faster training
-   **Comprehensive evaluation**: MAE, RMSE, MAPE, R², and bootstrap confidence intervals
-   **XAI integration**: GradCAM visualization for model interpretability

## Project Structure

```
HandOsteoNet/
├── Preprocessing/
│   ├── __init__.py
│   └── preprocessing.py          
├── Dataset/
│   ├── __init__.py
│   └── dataset.py               
├── Model/
│   ├── __init__.py
│   ├── cbam.py                  
│   ├── selfattention.py         
│   ├── dwconv.py                
│   ├── seg_part.py              
│   ├── predict_part.py          
│   └── model.py                 
├── Training/
│   ├── __init__.py
│   ├── loss_function.py         
│   ├── optimizer_scheduler.py   
│   └── train_val.py           
├── Evaluation/
│   ├── __init__.py
│   └── evaluation.py           
├── Utils/
│   ├── __init__.py
│   └── utils.py                
├── XAI/
│   ├── __init__.py
│   ├── gradcam.py              
│   ├── save_grad_images.py     
│   └── inverse_preprocessing.py 
├── main.py                     
└── README.md                   
```

## Installation

### Requirements

-   Python 3.8+
-   PyTorch 1.12+
-   torchvision
-   OpenCV
-   PIL
-   pandas
-   numpy
-   matplotlib
-   seaborn
-   tqdm
-   scikit-learn

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd HandOsteoNet
```

2. Install dependencies:

```bash
pip install torch torchvision opencv-python pillow pandas numpy matplotlib seaborn tqdm scikit-learn
```

## Data Preparation

### Dataset Structure

Prepare your dataset in the following format:

```
data/
├── train/
│   ├── 1.png
│   ├── 2.png
│   └── ...
├── val/
│   ├── 1001.png
│   ├── 1002.png
│   └── ...
├── test/
│   ├── 2001.png
│   ├── 2002.png
│   └── ...
├── train.csv
├── val.csv
└── test.csv
```

### CSV Format

Each CSV file should contain:

-   `id`: Image ID (corresponds to PNG filename)
-   `boneage`: Bone age in months
-   `male`: Gender (1 for male, 0 for female)

## Usage

### Training

Run the main training script:

```bash
python main.py \
    --train_csv path/to/train.csv \
    --val_csv path/to/val.csv \
    --test_csv path/to/test.csv \
    --train_img_dir path/to/train/images \
    --val_img_dir path/to/val/images \
    --test_img_dir path/to/test/images \
    --batch_size 16 \
    --num_epochs 150 \
    --lr 3e-4 \
    --weight_decay 1e-4 \
    --warmup_epochs 5 \
    --num_workers 4 \
    --seed 42 \
    --model_save_path best_bonenet.pth \
    --generate_gradcam
```

### Key Parameters

-   `--batch_size`: Batch size for training (default: 16)
-   `--num_epochs`: Number of training epochs (default: 150)
-   `--lr`: Learning rate (default: 3e-4)
-   `--weight_decay`: Weight decay for regularization (default: 1e-4)
-   `--warmup_epochs`: Number of warmup epochs (default: 5)
-   `--num_workers`: Number of data loader workers (default: 0)
-   `--seed`: Random seed for reproducibility (default: 42)
-   `--generate_gradcam`: Generate GradCAM visualizations after training

### Evaluation

The model automatically evaluates on the test set after training. Results are saved to:

-   `test_metrics.csv`: Comprehensive evaluation metrics
-   `predictions.csv`: Model predictions vs actual values
-   `training_metrics.csv`: Training history

### GradCAM Visualization

When `--generate_gradcam` is used, GradCAM images are saved to the `GradCAM/` directory. Each image shows:

-   Original hand radiograph (top)
-   GradCAM heatmap overlay (bottom)

## Model Performance

The HandOsteoNet model achieves:

-   **MAE**: ~4.70 ± 0.50 months
-   **MAPE**: ~4.72 ± 0.84%
-   **R²**: ~0.9766 
-   **MedAE**: ~3.70 ± 0.51 months

## Model Architecture Details

### Segmentation Module

-   **Encoder**: RegNet-Y-400MF (ImageNet pretrained)
-   **Decoder**: U-Net style with skip connections
-   **Output**: Single-channel segmentation mask

### Prediction Module

-   **Input**: 4-channel (3 RGB + 1 segmentation)
-   **Stem**: 3x3 conv with stride 2
-   **Stage 1**: Depthwise separable conv + CBAM + MaxPool
-   **Stage 2**: Depthwise separable conv + CBAM + MaxPool
-   **Stage 3**: Depthwise separable conv + Self-Attention + MaxPool
-   **Classifier**: FC layers with gender integration

### Attention Mechanisms

-   **CBAM**: Channel and spatial attention
-   **Self-Attention**: Multi-head self-attention for spatial relationships

## Preprocessing

The model uses advanced preprocessing:

1. **CLAHE Enhancement**: Two variants (4x4 and 8x8 grid)
2. **Multi-channel Input**: Original + CLAHE variants
3. **Data Augmentation**: Rotation, flip, crop during training
4. **Normalization**: ImageNet mean/std normalization

## Training Details

-   **Optimizer**: AdamW with weight decay
-   **Scheduler**: Warmup + Cosine Annealing
-   **Loss**: Combined Smooth L1 + MAE + MSE
-   **Mixed Precision**: Automatic mixed precision training
-   **Gradient Clipping**: Max norm of 5.0

## Output Files

After training, the following files are generated:

-   `best_bonenet.pth`: Best model weights
-   `training_metrics.csv`: Training history
-   `test_metrics.csv`: Test evaluation metrics
-   `predictions.csv`: Model predictions
-   `GradCAM/`: GradCAM visualization images

## License

Copyright (c) 2025 Qatar University. All rights reserved.

## Contact

For questions and support, please contact:

-   **Lead Researcher**: Dr. Amith Khandakar
-   **Institution**: Qatar University
-   **Email**: amitk@qu.edu.qa

## Acknowledgments

This work was supported by Qatar University and the research team led by Dr. Amith Khandakar.
