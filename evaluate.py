import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import io
import cv2
import random # Import random for seed setting

from dataset import create_dataloaders
from mobile_sam import sam_model_registry
from lora import MobileSAM_LoRA_Adapted
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
    
    # Use same seed as training for test dataloader consistency if needed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create data loaders
    _, _, test_loader = create_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed # Pass seed here too
    )
    
    # Load trained weights checkpoint
    checkpoint_path = Path(args.model_path)
    if not checkpoint_path.exists():
         print(f"Error: Model checkpoint not found at {args.model_path}")
         return 0.0 # Indicate failure
         
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load args used during training from checkpoint
    if 'args' not in checkpoint:
        print("Warning: Training args not found in checkpoint. Using defaults for model structure.")
        # Use provided args or defaults if necessary
        train_args = args 
    else:
         # Convert saved dict back to namespace
        train_args = argparse.Namespace(**checkpoint['args'])
        print("--- Model was trained with ---")
        for k, v in sorted(vars(train_args).items()):
            print(f"  {k}: {v}")
        print("-----------------------------")

    # Recreate model structure based on saved args
    model_type = "vit_t"
    base_model_test = sam_model_registry[model_type](checkpoint=None).to(device)
    
    # Determine if temp head was used based on train_decoder flag
    use_temp_head_eval = not train_args.train_decoder 
    
    eval_model = MobileSAM_LoRA_Adapted(
         model=base_model_test,
         r=train_args.lora_rank, 
         lora_alpha=train_args.lora_alpha,
         lora_dropout=0.0, # No dropout for eval
         train_encoder=train_args.train_encoder, # Structure needs flags
         train_decoder=train_args.train_decoder,
         use_temp_head=use_temp_head_eval 
    ).to(device)
    
    # Load the state dict
    eval_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded trained LoRA model state from epoch {checkpoint.get('epoch', 'N/A')}")
    eval_model.eval()

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
            
            # Forward pass with the loaded LoRA model
            outputs, _ = eval_model(images)
            
            if outputs is None:
                 print(f"Warning: Model output is None during evaluation (batch {batch_idx}). Skipping.")
                 continue
                 
            # Calculate metrics for the batch
            batch_metrics = calculate_metrics(outputs, masks, threshold=args.threshold)
            
            metrics['iou'].append(batch_metrics['IoU'])
            metrics['dice'].append(batch_metrics['Dice'])
            
            # Extract tissue type from image paths
            for i, img_path in enumerate(image_paths):
                tissue_type = Path(img_path).parent.parent.name # Get tissue folder name
                
                if tissue_type not in metrics['tissue_iou']:
                    metrics['tissue_iou'][tissue_type] = []
                    metrics['tissue_dice'][tissue_type] = []
                
                pred_mask_single = outputs[i:i+1]
                true_mask_single = masks[i:i+1]
                sample_metrics = calculate_metrics(pred_mask_single, true_mask_single, threshold=args.threshold)
                
                metrics['tissue_iou'][tissue_type].append(sample_metrics['IoU'])
                metrics['tissue_dice'][tissue_type].append(sample_metrics['Dice'])
                
                # Visualize if needed
                if args.visualize and (batch_idx * args.batch_size + i) < args.num_visualizations:
                    # Load original image for visualization (less processing)
                    orig_img_vis = cv2.imread(img_path)
                    orig_img_vis = cv2.cvtColor(orig_img_vis, cv2.COLOR_BGR2RGB)
                    
                    visualize_prediction(
                        image=orig_img_vis, # Use original image
                        true_mask=masks[i].cpu().numpy(),
                        pred_mask=torch.sigmoid(outputs[i, 0]).cpu().numpy(), # Pass probabilities
                        iou=sample_metrics['IoU'],
                        dice=sample_metrics['Dice'],
                        save_path=os.path.join(vis_dir, f"sample_{Path(img_path).stem}.png"),
                        tissue_type=tissue_type,
                        threshold=args.threshold # Pass threshold to viz
                    )
            
            # Update progress bar
            test_progress.set_postfix({
                'IoU': f"{batch_metrics['IoU']:.4f}",
                'Dice': f"{batch_metrics['Dice']:.4f}"
            })
    
    # Calculate average metrics
    mean_iou = np.mean(metrics['iou']) if metrics['iou'] else 0.0
    mean_dice = np.mean(metrics['dice']) if metrics['dice'] else 0.0
    
    # Calculate per-tissue metrics
    tissue_metrics_agg = {}
    print("\nMetrics by tissue type:")
    for tissue_type in sorted(metrics['tissue_iou'].keys()):
        tissue_iou_list = metrics['tissue_iou'][tissue_type]
        tissue_dice_list = metrics['tissue_dice'][tissue_type]
        num_samples = len(tissue_iou_list)
        avg_tissue_iou = np.mean(tissue_iou_list) if num_samples > 0 else 0.0
        avg_tissue_dice = np.mean(tissue_dice_list) if num_samples > 0 else 0.0
        tissue_metrics_agg[tissue_type] = {
            'iou': avg_tissue_iou,
            'dice': avg_tissue_dice,
            'num_samples': num_samples
        }
        print(f"  {tissue_type} ({num_samples} samples): IoU: {avg_tissue_iou:.4f}, Dice: {avg_tissue_dice:.4f}")
    
    # Save results
    results = {
        'overall': {
            'iou': mean_iou,
            'dice': mean_dice
        },
        'tissue_metrics': tissue_metrics_agg,
        'evaluation_args': vars(args) # Save eval args too
    }
    
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create a summary plot of performance by tissue type
    if args.plot and tissue_metrics_agg:
        plot_metrics_by_tissue(tissue_metrics_agg, os.path.join(args.output_dir, 'metrics_by_tissue.png'))
    
    return mean_dice


def visualize_prediction(image, true_mask, pred_mask, iou, dice, save_path, tissue_type, threshold=0.5):
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
        threshold: Threshold for converting prediction probabilities to binary masks
    """
    h, w = image.shape[:2]
    # Prediction mask already passed as probabilities, apply threshold here
    pred_mask_binary = (pred_mask > threshold).astype(np.uint8)
    true_mask_binary = (true_mask > 0).astype(np.uint8)
    
    # Colors
    green = np.array([0, 255, 0], dtype=np.uint8)
    red = np.array([255, 0, 0], dtype=np.uint8)
    blue = np.array([0, 0, 255], dtype=np.uint8) # For overlap
    
    # Create overlays
    alpha = 0.4
    overlay = image.copy()
    pred_overlay = image.copy()
    comparison = image.copy()
    
    true_idx = true_mask_binary > 0
    pred_idx = pred_mask_binary > 0
    overlap_idx = true_idx & pred_idx
    fp_idx = pred_idx & ~true_idx # False positive (Red)
    fn_idx = true_idx & ~pred_idx # False negative (Blue)
    
    # Green overlay for GT
    overlay[true_idx] = (alpha * green + (1 - alpha) * image[true_idx]).astype(np.uint8)
    # Red overlay for Pred
    pred_overlay[pred_idx] = (alpha * red + (1 - alpha) * image[pred_idx]).astype(np.uint8)
    # Comparison overlay
    comparison[fn_idx] = (alpha * blue + (1 - alpha) * image[fn_idx]).astype(np.uint8)
    comparison[fp_idx] = (alpha * red + (1 - alpha) * image[fp_idx]).astype(np.uint8)
    comparison[overlap_idx] = (alpha * green + (1 - alpha) * image[overlap_idx]).astype(np.uint8)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Tissue: {tissue_type.replace('_', ' ')} | IoU: {iou:.4f} | Dice: {dice:.4f}", fontsize=14)
    
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title("Ground Truth (Green)")
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(pred_overlay)
    axes[1, 0].set_title(f"Prediction (Red, Thresh={threshold:.2f})")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(comparison)
    axes[1, 1].set_title("Comparison (Green: TP, Red: FP, Blue: FN)")
    axes[1, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
    plt.savefig(save_path)
    plt.close(fig)


def plot_metrics_by_tissue(tissue_metrics, save_path):
    """
    Create a bar chart of performance by tissue type
    
    Args:
        tissue_metrics: Dictionary of metrics by tissue type
        save_path: Path to save the plot
    """
    if not tissue_metrics:
        print("No tissue metrics to plot.")
        return
        
    sorted_tissues = sorted(tissue_metrics.items(), key=lambda x: x[1]['dice'], reverse=True)
    tissue_names = [t[0] for t in sorted_tissues]
    dice_scores = [t[1]['dice'] for t in sorted_tissues]
    iou_scores = [t[1]['iou'] for t in sorted_tissues]
    sample_counts = [t[1]['num_samples'] for t in sorted_tissues]
    
    plt.figure(figsize=(max(10, len(tissue_names) * 0.5), 6))
    bar_width = 0.35
    index = np.arange(len(tissue_names))
    
    plt.bar(index, dice_scores, bar_width, label='Dice', color='skyblue')
    plt.bar(index + bar_width, iou_scores, bar_width, label='IoU', color='lightcoral')
    
    for i, count in enumerate(sample_counts):
        plt.text(i + bar_width / 2, max(dice_scores[i], iou_scores[i]) + 0.01, f"n={count}", ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Tissue Type', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance by Tissue Type', fontsize=14)
    plt.xticks(index + bar_width / 2, [name.replace('_', ' \n') for name in tissue_names], rotation=90, ha='center', fontsize=9)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned LoRA MobileSAM model')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='NuInsSeg', help='Path to NuInsSeg dataset')
    parser.add_argument('--image_size', type=int, default=1024, help='Input image size used during training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained LoRA model checkpoint (.pth file)')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Evaluation batch size')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for converting prediction probabilities to binary masks')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save evaluation results and visualizations')
    
    # Visualization parameters
    parser.add_argument('--visualize', action=argparse.BooleanOptionalAction, default=True, help='Generate visualizations of predictions vs ground truth')
    parser.add_argument('--num_visualizations', type=int, default=20, help='Maximum number of sample visualizations to save')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, default=True, help='Create summary plot of metrics by tissue type')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (for dataloader if applicable)')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate(args) 