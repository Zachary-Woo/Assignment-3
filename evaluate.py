import os
import torch
import torch.nn as nn
import argparse
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import io
import cv2

from dataset import create_dataloaders
from mobile_sam import setup_model
from lora import MobileSAM_LoRA
from train import calculate_metrics

def evaluate(args):
    """
    Evaluation function
    """
    # Create output directory
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualization directory if needed
    if args.visualize:
        vis_dir = save_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    _, _, test_loader = create_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    
    # Load pre-trained model
    model = setup_model()
    
    # Apply LoRA
    lora_model = MobileSAM_LoRA(
        model=model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,  # No dropout during evaluation
        train_encoder=True,  # Just to create the model structure, it's eval mode anyway
        train_decoder=True   # Just to create the model structure, it's eval mode anyway
    )
    
    # Load trained weights
    checkpoint = torch.load(args.model_path, map_location=device)
    lora_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {args.model_path}, epoch {checkpoint['epoch']}")
    
    # Print model info
    if 'args' in checkpoint:
        print("Model was trained with the following parameters:")
        for k, v in sorted(checkpoint['args'].items()):
            print(f"  {k}: {v}")
    
    # Move model to device
    lora_model = lora_model.to(device)
    
    # Set model to evaluation mode
    lora_model.eval()
    
    # Metrics
    metrics = {
        'iou': [],
        'dice': [],
        'tissue_iou': {},  # Store metrics by tissue type
        'tissue_dice': {}
    }
    
    # Process each batch
    with torch.no_grad():
        test_progress = tqdm(test_loader, desc="Evaluating")
        for batch_idx, batch in enumerate(test_progress):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            image_paths = batch['image_path']
            
            # Forward pass
            outputs, iou_preds = lora_model(images)
            
            # Calculate metrics
            batch_metrics = calculate_metrics(outputs, masks, threshold=args.threshold)
            
            # Store metrics
            metrics['iou'].append(batch_metrics['IoU'])
            metrics['dice'].append(batch_metrics['Dice'])
            
            # Extract tissue type from image paths
            for i, img_path in enumerate(image_paths):
                tissue_type = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
                
                if tissue_type not in metrics['tissue_iou']:
                    metrics['tissue_iou'][tissue_type] = []
                    metrics['tissue_dice'][tissue_type] = []
                
                # Calculate individual sample metrics
                pred_mask = outputs[i:i+1]
                true_mask = masks[i:i+1]
                sample_metrics = calculate_metrics(pred_mask, true_mask, threshold=args.threshold)
                
                metrics['tissue_iou'][tissue_type].append(sample_metrics['IoU'])
                metrics['tissue_dice'][tissue_type].append(sample_metrics['Dice'])
                
                # Visualize if needed
                if args.visualize and (batch_idx * args.batch_size + i) < args.num_visualizations:
                    visualize_prediction(
                        image=images[i].cpu().numpy(),
                        true_mask=masks[i].cpu().numpy(),
                        pred_mask=outputs[i, 0].cpu().numpy(),
                        iou=sample_metrics['IoU'],
                        dice=sample_metrics['Dice'],
                        save_path=os.path.join(vis_dir, f"sample_{batch_idx}_{i}.png"),
                        tissue_type=tissue_type
                    )
            
            # Update progress bar
            test_progress.set_postfix({
                'IoU': batch_metrics['IoU'],
                'Dice': batch_metrics['Dice']
            })
    
    # Calculate average metrics
    mean_iou = np.mean(metrics['iou'])
    mean_dice = np.mean(metrics['dice'])
    
    # Calculate per-tissue metrics
    tissue_metrics = {}
    for tissue_type in metrics['tissue_iou']:
        tissue_metrics[tissue_type] = {
            'iou': np.mean(metrics['tissue_iou'][tissue_type]),
            'dice': np.mean(metrics['tissue_dice'][tissue_type]),
            'num_samples': len(metrics['tissue_iou'][tissue_type])
        }
    
    # Print and save results
    print(f"Overall IoU: {mean_iou:.4f}")
    print(f"Overall Dice: {mean_dice:.4f}")
    
    # Print per-tissue metrics
    print("\nMetrics by tissue type:")
    for tissue_type, tissue_metric in sorted(tissue_metrics.items()):
        print(f"  {tissue_type} ({tissue_metric['num_samples']} samples):")
        print(f"    IoU: {tissue_metric['iou']:.4f}")
        print(f"    Dice: {tissue_metric['dice']:.4f}")
    
    # Save metrics to file
    results = {
        'overall': {
            'iou': mean_iou,
            'dice': mean_dice
        },
        'tissue_metrics': tissue_metrics
    }
    
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create a summary plot of performance by tissue type
    if args.plot:
        plot_metrics_by_tissue(tissue_metrics, os.path.join(args.output_dir, 'metrics_by_tissue.png'))
    
    return mean_dice


def visualize_prediction(image, true_mask, pred_mask, iou, dice, save_path, tissue_type):
    """
    Create visualization of prediction vs ground truth
    
    Args:
        image: Input image (C, H, W)
        true_mask: Ground truth mask (H, W)
        pred_mask: Predicted mask (H, W)
        iou: IoU score
        dice: Dice score
        save_path: Path to save visualization
        tissue_type: Type of tissue
    """
    # Convert image from CHW to HWC
    image = np.transpose(image, (1, 2, 0))
    
    # Normalize image for visualization
    image = (image - image.min()) / (image.max() - image.min())
    
    # Apply threshold to predicted mask
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    
    # Convert binary mask to RGB for visualization
    true_mask_rgb = np.zeros((true_mask.shape[0], true_mask.shape[1], 3), dtype=np.uint8)
    pred_mask_rgb = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    
    # Set colors: Green for true mask, Red for predicted mask
    true_mask_rgb[true_mask > 0] = [0, 255, 0]  # Green
    pred_mask_rgb[pred_mask > 0] = [255, 0, 0]  # Red
    
    # Create overlay
    overlay = image.copy()
    
    # Create alpha blending for true mask (green)
    alpha = 0.3
    idx = true_mask > 0
    overlay[idx] = alpha * np.array([0, 1, 0]) + (1 - alpha) * overlay[idx]
    
    # Create a separate overlay for the prediction
    pred_overlay = image.copy()
    idx = pred_mask > 0
    pred_overlay[idx] = alpha * np.array([1, 0, 0]) + (1 - alpha) * pred_overlay[idx]
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Set title with metrics
    plt.suptitle(f"Tissue: {tissue_type}, IoU: {iou:.4f}, Dice: {dice:.4f}", fontsize=16)
    
    # Original image
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')
    
    # Ground truth
    plt.subplot(2, 2, 2)
    plt.title("Ground Truth")
    plt.imshow(overlay)
    plt.axis('off')
    
    # Prediction
    plt.subplot(2, 2, 3)
    plt.title("Prediction")
    plt.imshow(pred_overlay)
    plt.axis('off')
    
    # Overlay both
    plt.subplot(2, 2, 4)
    plt.title("Comparison (Green: GT, Red: Pred)")
    
    # Create a separate comparison overlay
    comparison = image.copy()
    comparison[true_mask > 0] = alpha * np.array([0, 1, 0]) + (1 - alpha) * comparison[true_mask > 0]
    comparison[pred_mask > 0] = alpha * np.array([1, 0, 0]) + (1 - alpha) * comparison[pred_mask > 0]
    
    plt.imshow(comparison)
    plt.axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_metrics_by_tissue(tissue_metrics, save_path):
    """
    Create a bar chart of performance by tissue type
    
    Args:
        tissue_metrics: Dictionary of metrics by tissue type
        save_path: Path to save the plot
    """
    # Sort tissue types by dice score
    sorted_tissues = sorted(tissue_metrics.items(), key=lambda x: x[1]['dice'], reverse=True)
    
    tissue_names = [t[0] for t in sorted_tissues]
    dice_scores = [t[1]['dice'] for t in sorted_tissues]
    iou_scores = [t[1]['iou'] for t in sorted_tissues]
    sample_counts = [t[1]['num_samples'] for t in sorted_tissues]
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Define bar width
    bar_width = 0.35
    index = np.arange(len(tissue_names))
    
    # Create bars
    plt.bar(index, dice_scores, bar_width, label='Dice', color='blue')
    plt.bar(index + bar_width, iou_scores, bar_width, label='IoU', color='orange')
    
    # Add sample count on top of bars
    for i, count in enumerate(sample_counts):
        plt.text(i, dice_scores[i] + 0.01, f"{count}", ha='center', fontsize=8)
    
    # Add labels and title
    plt.xlabel('Tissue Type')
    plt.ylabel('Score')
    plt.title('Performance by Tissue Type')
    plt.xticks(index + bar_width / 2, [name.replace('_', ' ') for name in tissue_names], rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned model on NuInsSeg dataset')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='NuInsSeg', help='Path to NuInsSeg dataset')
    parser.add_argument('--image_size', type=int, default=1024, help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--lora_rank', type=int, default=4, help='Rank of LoRA adaptation')
    parser.add_argument('--lora_alpha', type=int, default=4, help='Alpha scaling factor for LoRA')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary segmentation')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save results')
    
    # Visualization parameters
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations of predictions')
    parser.add_argument('--num_visualizations', type=int, default=20, help='Number of samples to visualize')
    parser.add_argument('--plot', action='store_true', help='Create summary plots')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate(args) 