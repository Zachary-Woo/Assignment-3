import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path
import random

from mobile_sam import sam_model_registry
from lora import MobileSAM_LoRA_Adapted

def load_image(image_path, image_size=1024):
    """
    Load and preprocess an image for the model
    """
    # Load image
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize to target size (maintaining aspect ratio might be better for SAM)
    # For simplicity, using direct resize as before
    resized_image = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    
    # Normalize like the training script (using standard SAM normalization)
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    
    normalized_image = (resized_image - mean) / std
    
    # Convert to tensor (HWC to CHW) and add batch dimension
    tensor_image = torch.from_numpy(normalized_image.transpose(2, 0, 1)).float().unsqueeze(0)
    
    return tensor_image, image_rgb

def visualize_prediction(orig_image_rgb, prediction, output_path, threshold=0.5):
    """
    Visualize the model's prediction on an image
    """
    # Resize prediction to match original image size
    h, w = orig_image_rgb.shape[:2]
    pred_mask_binary = (prediction.squeeze().cpu().numpy() > threshold).astype(np.uint8)
    
    green = np.array([0, 255, 0], dtype=np.uint8)
    red = np.array([255, 0, 0], dtype=np.uint8)
    blue = np.array([0, 0, 255], dtype=np.uint8)
    
    alpha = 0.4
    pred_overlay = orig_image_rgb.copy()
    pred_idx = pred_mask_binary > 0
    pred_overlay[pred_idx] = (alpha * red + (1 - alpha) * orig_image_rgb[pred_idx]).astype(np.uint8)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Prediction (Threshold={threshold:.2f})")

    axes[0].imshow(orig_image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(pred_mask_binary, cmap='gray')
    axes[1].set_title("Predicted Mask")
    axes[1].axis('off')

    axes[2].imshow(pred_overlay)
    axes[2].set_title("Prediction Overlay (Red)")
    axes[2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close(fig)

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the trained LoRA model checkpoint
    checkpoint_path = Path(args.model_path)
    if not checkpoint_path.exists():
        print(f"Error: Trained model checkpoint not found at {args.model_path}")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loading trained model from: {args.model_path} (Epoch {checkpoint.get('epoch', 'N/A')})" )
    
    # Load args used during training
    if 'args' not in checkpoint:
        print("Warning: Training args not found in checkpoint. Using demo defaults for model structure.")
        train_args = args # Fallback, might be incorrect for structure
    else:
        train_args = argparse.Namespace(**checkpoint['args'])
        print("--- Model trained with ---")
        for k, v in sorted(vars(train_args).items()):
            print(f"  {k}: {v}")
        print("-------------------------")

    # Recreate model structure
    model_type = "vit_t"
    base_model = sam_model_registry[model_type](checkpoint=None).to(device) # Load architecture
    use_temp_head_demo = not train_args.train_decoder
    
    lora_model = MobileSAM_LoRA_Adapted(
         model=base_model,
         r=train_args.lora_rank,
         lora_alpha=train_args.lora_alpha,
         lora_dropout=0.0,
         train_encoder=train_args.train_encoder,
         train_decoder=train_args.train_decoder,
         use_temp_head=use_temp_head_demo
    ).to(device)
    
    # Load state dict
    lora_model.load_state_dict(checkpoint['model_state_dict'])
    lora_model.eval()
    print("Successfully loaded LoRA model state.")

    # Find all tissue image directories
    tissue_image_dirs = []
    for tissue_folder in os.listdir(args.data_dir):
        tissue_path = os.path.join(args.data_dir, tissue_folder)
        if os.path.isdir(tissue_path):
            img_dir = os.path.join(tissue_path, 'tissue images')
            if os.path.exists(img_dir):
                tissue_image_dirs.append(img_dir)
    
    if not tissue_image_dirs:
        print(f"No tissue image directories found in {args.data_dir}")
        return
    
    # Select random sample images
    all_images = []
    for img_dir in tissue_image_dirs:
        image_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        tissue_type = Path(img_dir).parent.name
        all_images.extend([(os.path.join(img_dir, img), tissue_type) for img in image_files])
    
    # Ensure we have enough images
    if len(all_images) < args.num_samples:
        print(f"Warning: Only {len(all_images)} images available, using all.")
        sample_images = all_images
    else:
        sample_images = random.sample(all_images, args.num_samples)
    
    # Process each sample image
    for i, (image_path, tissue_type) in enumerate(sample_images):
        print(f"Processing {i+1}/{len(sample_images)}: {Path(image_path).name} ({tissue_type})")
        
        # Load and preprocess image using SAM's preprocessing
        input_image_tensor, original_image_rgb = load_image(image_path, args.image_size)
        input_image_tensor = input_image_tensor.to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs, _ = lora_model(input_image_tensor)
        
        if outputs is None:
             print("Skipping visualization due to None output.")
             continue
        
        # Visualize prediction
        output_filename = f"{tissue_type}_{Path(image_path).stem}_pred.png"
        output_path = os.path.join(args.output_dir, output_filename)
        
        # Pass probabilities to visualization
        visualize_prediction(original_image_rgb, torch.sigmoid(outputs[0, 0]), output_path, args.threshold)
        
        print(f"Saved prediction visualization to {output_path}")
    
    print(f"\nDemo finished. {len(sample_images)} predictions saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo for LoRA fine-tuned MobileSAM Nuclei Segmentation")
    # Model parameters - Expects a checkpoint from train.py
    parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned LoRA model checkpoint (.pth)')
    # Lora rank/alpha loaded from checkpoint, no need for args here usually
    # parser.add_argument('--lora_rank', type=int, help='Rank used during training (loaded from checkpoint)')
    # parser.add_argument('--lora_alpha', type=int, help='Alpha used during training (loaded from checkpoint)')
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='NuInsSeg', help='Path to NuInsSeg dataset root directory')
    parser.add_argument('--image_size', type=int, default=1024, help='Input image size model was trained with')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of random samples to process for the demo')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for converting prediction probabilities to binary masks')
    parser.add_argument('--output_dir', type=str, default='demo_outputs', help='Directory to save demo visualization outputs')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    
    args = parser.parse_args()
    main(args) 