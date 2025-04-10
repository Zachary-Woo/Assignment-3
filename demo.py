import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path
import random

from mobile_sam import setup_model
from lora import MobileSAM_LoRA

def load_image(image_path, image_size=1024):
    """
    Load and preprocess an image for the model
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to target size
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image.astype(np.float32) / 255.0
    image = (image - mean) / std
    
    # Convert to tensor and add batch dimension
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    
    return image

def visualize_prediction(image_path, prediction, output_path, threshold=0.5):
    """
    Visualize the model's prediction on an image
    """
    # Load original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize prediction to match original image size
    h, w = image.shape[:2]
    pred_mask = prediction.squeeze().cpu().numpy()
    pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Apply threshold
    pred_mask = (pred_mask > threshold).astype(np.uint8)
    
    # Create colored mask for visualization
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    colored_mask[pred_mask > 0] = [0, 255, 0]  # Green for nuclei
    
    # Create alpha blending
    alpha = 0.5
    overlay = image.copy()
    mask = pred_mask > 0
    overlay[mask] = (alpha * np.array([0, 255, 0]) + (1 - alpha) * image[mask]).astype(np.uint8)
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')
    
    # Prediction mask
    plt.subplot(1, 3, 2)
    plt.title("Predicted Nuclei")
    plt.imshow(pred_mask, cmap='gray')
    plt.axis('off')
    
    # Overlay
    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load base model
    model = setup_model()
    
    # Apply LoRA
    lora_model = MobileSAM_LoRA(
        model=model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,  # No dropout for inference
        train_encoder=True,
        train_decoder=True
    )
    
    # Load trained weights
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location=device)
        lora_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded fine-tuned model from {args.model_path}")
    else:
        print("Using model without fine-tuning")
    
    # Move model to device
    lora_model = lora_model.to(device)
    lora_model.eval()
    
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
        tissue_type = os.path.basename(os.path.dirname(img_dir))
        all_images.extend([(os.path.join(img_dir, img), tissue_type) for img in image_files])
    
    # Ensure we have enough images
    if len(all_images) < args.num_samples:
        print(f"Warning: Only {len(all_images)} images available, less than requested {args.num_samples}")
        sample_images = all_images
    else:
        sample_images = random.sample(all_images, args.num_samples)
    
    # Process each sample image
    for i, (image_path, tissue_type) in enumerate(sample_images):
        print(f"Processing image {i+1}/{len(sample_images)}: {image_path}")
        
        # Load image
        image = load_image(image_path, args.image_size)
        image = image.to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs, _ = lora_model(image)
        
        # Visualize prediction
        output_path = os.path.join(args.output_dir, f"{tissue_type}_{os.path.basename(image_path).replace('.png', '_pred.png')}")
        visualize_prediction(image_path, outputs[:, 0], output_path, args.threshold)
        
        print(f"Saved prediction to {output_path}")
    
    print(f"Done! {len(sample_images)} predictions saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo for nuclei instance segmentation")
    
    # Model parameters
    parser.add_argument('--model_path', type=str, default=None, help='Path to fine-tuned model checkpoint')
    parser.add_argument('--lora_rank', type=int, default=4, help='Rank of LoRA adaptation')
    parser.add_argument('--lora_alpha', type=int, default=4, help='Alpha scaling factor for LoRA')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='NuInsSeg', help='Path to NuInsSeg dataset')
    parser.add_argument('--image_size', type=int, default=1024, help='Input image size')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of random samples to process')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary segmentation')
    parser.add_argument('--output_dir', type=str, default='demo_outputs', help='Directory to save outputs')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    
    args = parser.parse_args()
    main(args) 