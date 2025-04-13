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
from skimage import measure # For connected components
from scipy.optimize import linear_sum_assignment # For solving assignment problem in PQ

from dataset import create_dataloaders
from mobile_sam import sam_model_registry
from lora import MobileSAM_LoRA_Adapted
from train import calculate_metrics

# Add instance segmentation metrics
def calculate_aji(pred_mask, true_mask):
    """
    Calculate Aggregated Jaccard Index (AJI) for instance segmentation.
    
    Args:
        pred_mask: Binary prediction mask (H, W)
        true_mask: Ground truth instance mask (H, W) with unique IDs for each instance
    
    Returns:
        AJI score (float)
    """
    # Create instance segmentation from binary pred_mask using connected components
    if np.max(pred_mask) <= 1:  # If it's a binary mask
        labeled_pred = measure.label(pred_mask)
    else:
        labeled_pred = pred_mask.copy()
        
    # Ground truth instances
    if np.max(true_mask) <= 1:  # If it's a binary mask (shouldn't be the case)
        print("Warning: Ground truth mask appears to be binary, not instance segmentation")
        labeled_true = measure.label(true_mask)
    else:
        labeled_true = true_mask.copy()
    
    # Get unique instances (exclude background 0)
    true_ids = np.unique(labeled_true)[1:]  # Skip background
    pred_ids = np.unique(labeled_pred)[1:]  # Skip background
    
    if len(true_ids) == 0 or len(pred_ids) == 0:
        if len(true_ids) == len(pred_ids):  # Both empty
            return 1.0
        else:  # One is empty, the other is not
            return 0.0
    
    # Compute IoU between each pair of predicted and true instances
    iou_matrix = np.zeros((len(true_ids), len(pred_ids)))
    
    for i, true_id in enumerate(true_ids):
        true_mask_i = (labeled_true == true_id)
        for j, pred_id in enumerate(pred_ids):
            pred_mask_j = (labeled_pred == pred_id)
            
            # Calculate IoU
            intersection = np.sum(np.logical_and(true_mask_i, pred_mask_j))
            union = np.sum(np.logical_or(true_mask_i, pred_mask_j))
            
            iou_matrix[i, j] = intersection / union if union > 0 else 0.0
    
    # Find the best matching using the Hungarian algorithm
    true_indices, pred_indices = linear_sum_assignment(-iou_matrix)
    
    # Calculate AJI
    numerator = 0
    denominator = 0
    
    used_pred = set()
    for i, j in zip(true_indices, pred_indices):
        if iou_matrix[i, j] > 0:
            true_id = true_ids[i]
            pred_id = pred_ids[j]
            
            true_mask_i = (labeled_true == true_id)
            pred_mask_j = (labeled_pred == pred_id)
            
            intersection = np.sum(np.logical_and(true_mask_i, pred_mask_j))
            union = np.sum(np.logical_or(true_mask_i, pred_mask_j))
            
            numerator += intersection
            denominator += union
            
            used_pred.add(pred_id)
    
    # Add the remaining predictions to the denominator
    for pred_id in pred_ids:
        if pred_id not in used_pred:
            pred_mask_j = (labeled_pred == pred_id)
            denominator += np.sum(pred_mask_j)
    
    aji = numerator / denominator if denominator > 0 else 0.0
    return aji

def calculate_pq(pred_mask, true_mask, iou_threshold=0.5):
    """
    Calculate Panoptic Quality (PQ) for instance segmentation.
    
    Args:
        pred_mask: Binary prediction mask (H, W)
        true_mask: Ground truth instance mask (H, W) with unique IDs for each instance
        iou_threshold: IoU threshold for true positive determination
    
    Returns:
        PQ score (float), with components SQ (Segmentation Quality) and RQ (Recognition Quality)
    """
    # Create instance segmentation from binary pred_mask using connected components
    if np.max(pred_mask) <= 1:  # If it's a binary mask
        labeled_pred = measure.label(pred_mask)
    else:
        labeled_pred = pred_mask.copy()
    
    # Ground truth instances
    if np.max(true_mask) <= 1:  # If it's a binary mask (shouldn't be the case)
        labeled_true = measure.label(true_mask)
    else:
        labeled_true = true_mask.copy()
    
    # Get unique instances (exclude background 0)
    true_ids = np.unique(labeled_true)[1:]  # Skip background
    pred_ids = np.unique(labeled_pred)[1:]  # Skip background
    
    if len(true_ids) == 0 and len(pred_ids) == 0:  # Both empty
        return {'PQ': 1.0, 'SQ': 1.0, 'RQ': 1.0}
    
    if len(true_ids) == 0 or len(pred_ids) == 0:  # One is empty, the other is not
        return {'PQ': 0.0, 'SQ': 0.0, 'RQ': 0.0}
    
    # Compute IoU between each pair of predicted and true instances
    iou_matrix = np.zeros((len(true_ids), len(pred_ids)))
    
    for i, true_id in enumerate(true_ids):
        true_mask_i = (labeled_true == true_id)
        for j, pred_id in enumerate(pred_ids):
            pred_mask_j = (labeled_pred == pred_id)
            
            # Calculate IoU
            intersection = np.sum(np.logical_and(true_mask_i, pred_mask_j))
            union = np.sum(np.logical_or(true_mask_i, pred_mask_j))
            
            iou_matrix[i, j] = intersection / union if union > 0 else 0.0
    
    # Find matches above threshold
    true_matches = -np.ones(len(true_ids), dtype=int)
    pred_matches = -np.ones(len(pred_ids), dtype=int)
    
    # Greedy matching based on IoU
    for i in range(len(true_ids)):
        for j in range(len(pred_ids)):
            if iou_matrix[i, j] >= iou_threshold:
                if true_matches[i] == -1 and pred_matches[j] == -1:
                    true_matches[i] = j
                    pred_matches[j] = i
    
    # Count statistics
    tp = np.sum(true_matches >= 0)
    fp = len(pred_ids) - tp
    fn = len(true_ids) - tp
    
    # Calculate PQ metrics
    if tp == 0:
        return {'PQ': 0.0, 'SQ': 0.0, 'RQ': 0.0}
    
    # Segmentation quality: average IoU of matched segments
    sq = np.mean([iou_matrix[i, true_matches[i]] for i in range(len(true_ids)) if true_matches[i] >= 0])
    
    # Recognition quality: F1 score
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + 0.5 * fp + 0.5 * fn) > 0 else 0.0
    
    # Panoptic quality
    pq = sq * rq
    
    return {'PQ': pq, 'SQ': sq, 'RQ': rq}

def main_evaluate(args):
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
        'aji': [],      # Added AJI metric
        'pq': [],       # Added PQ metric
        'sq': [],       # Added SQ component of PQ
        'rq': [],       # Added RQ component of PQ
        'tissue_iou': {},  # Store metrics by tissue type
        'tissue_dice': {},
        'tissue_aji': {},  # Added tissue-specific AJI
        'tissue_pq': {}    # Added tissue-specific PQ
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
                    metrics['tissue_aji'][tissue_type] = []
                    metrics['tissue_pq'][tissue_type] = []
                
                pred_mask_single = outputs[i:i+1]
                true_mask_single = masks[i:i+1]
                sample_metrics = calculate_metrics(pred_mask_single, true_mask_single, threshold=args.threshold)
                
                metrics['tissue_iou'][tissue_type].append(sample_metrics['IoU'])
                metrics['tissue_dice'][tissue_type].append(sample_metrics['Dice'])
                
                # Calculate instance metrics for this sample
                # Convert tensors to numpy for instance metrics
                binary_pred = (torch.sigmoid(outputs[i, 0]) > args.threshold).cpu().numpy().astype(np.uint8)
                true_mask_np = true_mask_single[0].cpu().numpy()
                
                # Calculate AJI
                aji_score = calculate_aji(binary_pred, true_mask_np)
                metrics['aji'].append(aji_score)
                metrics['tissue_aji'][tissue_type].append(aji_score)
                
                # Calculate PQ
                pq_results = calculate_pq(binary_pred, true_mask_np, iou_threshold=0.5)
                metrics['pq'].append(pq_results['PQ'])
                metrics['sq'].append(pq_results['SQ'])
                metrics['rq'].append(pq_results['RQ'])
                metrics['tissue_pq'][tissue_type].append(pq_results['PQ'])
                
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
                        aji=aji_score,
                        pq=pq_results['PQ'],
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
    mean_aji = np.mean(metrics['aji']) if metrics['aji'] else 0.0
    mean_pq = np.mean(metrics['pq']) if metrics['pq'] else 0.0
    mean_sq = np.mean(metrics['sq']) if metrics['sq'] else 0.0
    mean_rq = np.mean(metrics['rq']) if metrics['rq'] else 0.0
    
    # Calculate per-tissue metrics
    tissue_metrics_agg = {}
    print("\nMetrics by tissue type:")
    for tissue_type in sorted(metrics['tissue_iou'].keys()):
        tissue_iou_list = metrics['tissue_iou'][tissue_type]
        tissue_dice_list = metrics['tissue_dice'][tissue_type]
        tissue_aji_list = metrics['tissue_aji'][tissue_type]
        tissue_pq_list = metrics['tissue_pq'][tissue_type]
        num_samples = len(tissue_iou_list)
        avg_tissue_iou = np.mean(tissue_iou_list) if num_samples > 0 else 0.0
        avg_tissue_dice = np.mean(tissue_dice_list) if num_samples > 0 else 0.0
        avg_tissue_aji = np.mean(tissue_aji_list) if num_samples > 0 else 0.0
        avg_tissue_pq = np.mean(tissue_pq_list) if num_samples > 0 else 0.0
        tissue_metrics_agg[tissue_type] = {
            'iou': avg_tissue_iou,
            'dice': avg_tissue_dice,
            'aji': avg_tissue_aji,
            'pq': avg_tissue_pq,
            'num_samples': num_samples
        }
        print(f"  {tissue_type} ({num_samples} samples): IoU: {avg_tissue_iou:.4f}, Dice: {avg_tissue_dice:.4f}, AJI: {avg_tissue_aji:.4f}, PQ: {avg_tissue_pq:.4f}")
    
    # Save results
    results = {
        'overall': {
            'iou': mean_iou,
            'dice': mean_dice,
            'aji': mean_aji,
            'pq': mean_pq,
            'sq': mean_sq,
            'rq': mean_rq
        },
        'tissue_metrics': tissue_metrics_agg,
        'evaluation_args': vars(args) # Save eval args too
    }
    
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create a summary plot of performance by tissue type
    if args.plot and tissue_metrics_agg:
        plot_metrics_by_tissue(tissue_metrics_agg, os.path.join(args.output_dir, 'metrics_by_tissue.png'))
    
    print(f"\nOverall Results:")
    print(f"  IoU:  {mean_iou:.4f}")
    print(f"  Dice: {mean_dice:.4f}")
    print(f"  AJI:  {mean_aji:.4f}")
    print(f"  PQ:   {mean_pq:.4f}")
    print(f"  SQ:   {mean_sq:.4f}")
    print(f"  RQ:   {mean_rq:.4f}")
    
    return mean_dice


def visualize_prediction(image, true_mask, pred_mask, iou, dice, aji, pq, save_path, tissue_type, threshold=0.5):
    """
    Create visualization of prediction vs ground truth
    
    Args:
        image: Input image (H, W, C)
        true_mask: Ground truth mask (H, W)
        pred_mask: Predicted mask (H, W)
        iou: IoU score
        dice: Dice score
        aji: AJI score
        pq: PQ score
        save_path: Path to save visualization
        tissue_type: Type of tissue
        threshold: Threshold for converting prediction probabilities to binary masks
    """
    # Make sure image and masks have same dimensions
    h, w = image.shape[:2]
    
    # Resize masks to match image dimensions if needed
    if true_mask.shape[0] != h or true_mask.shape[1] != w:
        true_mask = cv2.resize(true_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    if pred_mask.shape[0] != h or pred_mask.shape[1] != w:
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
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
    
    # Create boolean masks for overlays
    true_idx = np.where(true_mask_binary > 0)
    pred_idx = np.where(pred_mask_binary > 0)
    
    # Create masks for comparison
    true_mask_2d = true_mask_binary > 0
    pred_mask_2d = pred_mask_binary > 0
    overlap_mask = true_mask_2d & pred_mask_2d
    fp_mask = pred_mask_2d & ~true_mask_2d  # False positive
    fn_mask = true_mask_2d & ~pred_mask_2d  # False negative
    
    # Apply overlays - safe indexing
    for i, j in zip(*true_idx):
        overlay[i, j] = (alpha * green + (1 - alpha) * image[i, j]).astype(np.uint8)
    
    for i, j in zip(*pred_idx):
        pred_overlay[i, j] = (alpha * red + (1 - alpha) * image[i, j]).astype(np.uint8)
    
    # Comparison overlay
    fn_idx = np.where(fn_mask)
    fp_idx = np.where(fp_mask)
    overlap_idx = np.where(overlap_mask)
    
    for i, j in zip(*fn_idx):
        comparison[i, j] = (alpha * blue + (1 - alpha) * image[i, j]).astype(np.uint8)
    
    for i, j in zip(*fp_idx):
        comparison[i, j] = (alpha * red + (1 - alpha) * image[i, j]).astype(np.uint8)
    
    for i, j in zip(*overlap_idx):
        comparison[i, j] = (alpha * green + (1 - alpha) * image[i, j]).astype(np.uint8)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Tissue: {tissue_type.replace('_', ' ')}\nIoU: {iou:.4f}, Dice: {dice:.4f}, AJI: {aji:.4f}, PQ: {pq:.4f}", fontsize=14)
    
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
    aji_scores = [t[1]['aji'] for t in sorted_tissues]
    pq_scores = [t[1]['pq'] for t in sorted_tissues]
    sample_counts = [t[1]['num_samples'] for t in sorted_tissues]
    
    # Create a figure with subplots for different metrics
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(tissue_names) * 0.5), 12))
    
    # Plot Dice and IoU in the first subplot
    bar_width = 0.25
    index = np.arange(len(tissue_names))
    
    axes[0].bar(index - bar_width/2, dice_scores, bar_width, label='Dice', color='skyblue')
    axes[0].bar(index + bar_width/2, iou_scores, bar_width, label='IoU', color='lightcoral')
    
    for i, count in enumerate(sample_counts):
        axes[0].text(i, max(dice_scores[i], iou_scores[i]) + 0.01, f"n={count}", ha='center', va='bottom', fontsize=8)
    
    axes[0].set_xlabel('Tissue Type', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Dice and IoU by Tissue Type', fontsize=14)
    axes[0].set_xticks(index)
    axes[0].set_xticklabels([name.replace('_', ' \n') for name in tissue_names], rotation=90, ha='center', fontsize=9)
    axes[0].set_yticks(np.arange(0, 1.1, 0.1))
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc='lower right')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot AJI and PQ in the second subplot
    axes[1].bar(index - bar_width/2, aji_scores, bar_width, label='AJI', color='lightgreen')
    axes[1].bar(index + bar_width/2, pq_scores, bar_width, label='PQ', color='plum')
    
    for i, count in enumerate(sample_counts):
        axes[1].text(i, max(aji_scores[i], pq_scores[i]) + 0.01, f"n={count}", ha='center', va='bottom', fontsize=8)
    
    axes[1].set_xlabel('Tissue Type', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('AJI and PQ by Tissue Type', fontsize=14)
    axes[1].set_xticks(index)
    axes[1].set_xticklabels([name.replace('_', ' \n') for name in tissue_names], rotation=90, ha='center', fontsize=9)
    axes[1].set_yticks(np.arange(0, 1.1, 0.1))
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc='lower right')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    
    # Also create a separate file with the average metric values
    plt.figure(figsize=(10, 6))
    metrics_names = ['Dice', 'IoU', 'AJI', 'PQ']
    metrics_values = [np.mean(dice_scores), np.mean(iou_scores), np.mean(aji_scores), np.mean(pq_scores)]
    
    plt.bar(metrics_names, metrics_values, color=['skyblue', 'lightcoral', 'lightgreen', 'plum'])
    plt.axhline(y=np.mean(metrics_values), color='gray', linestyle='--', alpha=0.7)
    
    plt.ylim(0, 1.0)
    plt.title('Average Metric Performance', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add value labels on top of bars
    for i, v in enumerate(metrics_values):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontsize=10)
        
    plot_dir = os.path.dirname(save_path)
    plt.savefig(os.path.join(plot_dir, 'average_metrics.png'), dpi=150)
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
    main_evaluate(args) 