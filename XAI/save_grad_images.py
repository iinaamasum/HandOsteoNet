import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from .gradcam import GradCAM

def inverse_normalize():
    """Get inverse normalization transform"""
    return transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

def save_gradcam_image(img_tensor, cam, filename):
    """
    Save GradCAM visualization image
    
    Args:
        img_tensor: Input image tensor
        cam: GradCAM heatmap
        filename: Output filename
    """
    inv_normalize = inverse_normalize()
    img_tensor = inv_normalize(img_tensor)
    img_np = img_tensor.detach().permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)

    orig_img = img_np[:, :, 0]

    cam_resized = cv2.resize(cam, (480, 480), interpolation=cv2.INTER_LINEAR)
    cam_resized = np.clip(cam_resized, 0, 1)

    heatmap = plt.get_cmap('jet')(cam_resized)[:, :, :3]
    overlayed = np.clip(heatmap * 0.4 + img_np * 0.6, 0, 1)

    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB)

    merged = np.concatenate([orig_img_rgb, overlayed], axis=0)
    merged = (merged * 255).astype(np.uint8)

    Image.fromarray(merged).save(filename)

def generate_gradcam_images(model, test_loader, device, output_dir="GradCAM"):
    """
    Generate GradCAM images for test dataset
    
    Args:
        model: The trained model
        test_loader: Test data loader
        device: Device to run on
        output_dir: Directory to save GradCAM images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    target_layer = model.seg_model.s4
    gradcam = GradCAM(model, target_layer)

    model.eval()
    count = 0
    saved_images = []
    
    for images, boneages, genders, img_ids in tqdm(test_loader, desc="Generating GradCAM"):
        images = images.to(device)
        genders = genders.to(device).float()
        images.requires_grad_(True)

        cams = gradcam(images, genders)

        for i in range(images.size(0)):
            filename = f"{output_dir}/gradcam_{img_ids[i]}.png"
            save_gradcam_image(images[i].cpu(), cams[i], filename)
            saved_images.append((img_ids[i], filename))
            count += 1

    gradcam.remove_hooks()
    print(f"Generated {count} Grad-CAM images saved in '{output_dir}' directory.")
    
    return saved_images