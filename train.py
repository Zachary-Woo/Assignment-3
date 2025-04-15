import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
import random
from tqdm import tqdm
import json
import time
from pathlib import Path

from dataset import create_dataloaders, create_cross_val_indices
# Use the official MobileSAM registry
from mobile_sam import sam_model_registry
# Import the adapted LoRA wrapper
from lora import MobileSAM_LoRA_Adapted

# Loss functions
class DiceLoss(nn.Module):
    """
    Implementation of the Dice Loss, a region-based metric commonly used for medical image segmentation.
    
    The Dice coefficient quantifies the spatial overlap between two binary volumes, and this loss
    represents 1 - Dice coefficient to be minimized during training.
    """
    def __init__(self, smooth=1.0):
        """
        Initialize Dice Loss with a smoothing factor to prevent division by zero.
        
        Args:
            smooth: Small constant added to numerator and denominator to prevent numerical instability
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Compute the Dice Loss between predicted and target segmentation masks.
        
        Args:
            pred: Model predictions (B, C, H, W)
            target: Ground truth masks (B, H, W)
            
        Returns:
            Dice Loss value (1 - Dice coefficient)
        """
        binary_target = (target > 0).float()
        # Error handling for null predictions
        if pred is None:
             print("Warning: DiceLoss received None prediction.")
             return torch.tensor(1.0, device=target.device) # Return maximum loss
             
        # Process prediction based on channel dimension
        if pred.shape[1] > 1:
            # Extract foreground channel if multi-channel output
            pred = torch.sigmoid(pred[:, 0, :, :]) 
        else:
            pred = torch.sigmoid(pred.squeeze(1))
            
        # Flatten spatial dimensions for computation
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = binary_target.view(target.size(0), -1)
        
        # Calculate intersection and union for Dice coefficient
        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1)
        
        # Compute Dice coefficient with smoothing factor
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return (1 - dice).mean() # Return batch-averaged Dice Loss

class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss, designed to address class imbalance by down-weighting well-classified examples.
    
    Focal Loss modifies standard cross-entropy by adding a factor that reduces the loss contribution
    from easy examples and increases the importance of correcting misclassified examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Initialize Focal Loss with focusing parameters.
        
        Args:
            alpha: Weighting factor for the rare class (typically foreground)
            gamma: Focusing parameter that adjusts the rate at which easy examples are down-weighted
            reduction: Method to reduce loss over batch ('mean', 'sum', or 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Compute the Focal Loss between predicted and target segmentation masks.
        
        Args:
            pred: Model predictions (B, C, H, W)
            target: Ground truth masks (B, H, W)
            
        Returns:
            Focal Loss value
        """
        binary_target = (target > 0).float()
        
        # Error handling for null predictions
        if pred is None:
             print("Warning: FocalLoss received None prediction.")
             return torch.tensor(10.0, device=target.device) 

        # Process prediction based on channel dimension
        if pred.shape[1] > 1:
            pred = torch.sigmoid(pred[:, 0, :, :]) 
        else:
            pred = torch.sigmoid(pred.squeeze(1))
            
        # Flatten spatial dimensions for computation
        binary_target = binary_target.view(-1)
        pred = pred.view(-1)
        
        # Numerical stability: prevent logarithm of zero
        eps = 1e-8
        pred = torch.clamp(pred, eps, 1. - eps)
        
        # Compute probabilities for the given class
        pt = pred * binary_target + (1 - pred) * (1 - binary_target)
        # Apply alpha balancing
        alpha_factor = self.alpha * binary_target + (1 - self.alpha) * (1 - binary_target)
        # Apply focusing factor (1-pt)^gamma
        modulating_factor = (1.0 - pt) ** self.gamma
        
        # Calculate focal loss using binary cross entropy as base
        bce_loss = F.binary_cross_entropy(pred, binary_target, reduction='none')
        loss = alpha_factor * modulating_factor * bce_loss
        
        # Apply specified reduction method
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class CombinedLoss(nn.Module):
    """
    Weighted combination of Dice Loss and Focal Loss for improved segmentation performance.
    
    This combined approach balances the region-based properties of Dice Loss with
    the class-balanced pixel-wise focus of Focal Loss.
    """
    def __init__(self, dice_weight=0.5, focal_weight=0.5, smooth=1.0, alpha=0.25, gamma=2.0):
        """
        Initialize Combined Loss with weighting parameters.
        
        Args:
            dice_weight: Weight assigned to Dice Loss component
            focal_weight: Weight assigned to Focal Loss component
            smooth: Smoothing factor for Dice Loss
            alpha: Class balancing factor for Focal Loss
            gamma: Focusing parameter for Focal Loss
        """
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(smooth)
        self.focal_loss = FocalLoss(alpha, gamma)
    
    def forward(self, pred, target):
        """
        Compute the weighted combination of Dice and Focal losses.
        
        Args:
            pred: Model predictions (B, C, H, W)
            target: Ground truth masks (B, H, W)
            
        Returns:
            Weighted sum of Dice and Focal losses
        """
        if pred is None:
             print("Warning: CombinedLoss received None prediction.")
             return torch.tensor(11.0, device=target.device) 
             
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.dice_weight * dice + self.focal_weight * focal

# Evaluation metrics
def calculate_metrics(pred, target, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) and Dice coefficient metrics for segmentation evaluation.
    
    Args:
        pred: Model predictions (B, C, H, W)
        target: Ground truth masks (B, H, W)
        threshold: Probability threshold for converting predictions to binary masks
        
    Returns:
        Dictionary containing IoU and Dice metrics
    """
    if pred is None or target is None:
        return {'IoU': 0.0, 'Dice': 0.0}
    
    with torch.no_grad(): # Disable gradient computation for efficiency
         # Extract appropriate channel if multi-channel output
         if pred.shape[1] > 1:
             pred = pred[:, 0, :, :] # Select first channel
         else:
             pred = pred.squeeze(1)
         
         # Apply sigmoid and threshold to get binary predictions
         pred = (torch.sigmoid(pred) > threshold).float()
         binary_target = (target > 0).float()
         
         # Flatten spatial dimensions for computation
         pred_flat = pred.view(pred.size(0), -1)
         target_flat = binary_target.view(target.size(0), -1)
         
         # Calculate intersection, union, and sums for metrics
         intersection = (pred_flat * target_flat).sum(1)
         pred_sum = pred_flat.sum(1)
         target_sum = target_flat.sum(1)
         union = pred_sum + target_sum - intersection
         
         # Calculate IoU and Dice with small epsilon for numerical stability
         iou = (intersection + 1e-8) / (union + 1e-8)
         dice = (2 * intersection + 1e-8) / (pred_sum + target_sum + 1e-8)
         
         # Handle edge cases where both prediction and target are empty
         iou[union == 0] = 1.0
         dice[pred_sum + target_sum == 0] = 1.0
         
         # Average metrics across batch
         mean_iou = iou.mean().item()
         mean_dice = dice.mean().item()
         
    return {'IoU': mean_iou, 'Dice': mean_dice}

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs, grad_clip=1.0):
    model.train()
    epoch_loss = 0.0
    epoch_metrics = {'IoU': 0.0, 'Dice': 0.0}
    num_batches = len(train_loader)
    train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]")
    
    for batch in train_progress:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        outputs, _ = model(images) # Forward pass using the LoRA wrapped model
        
        if outputs is None:
            print(f"Warning: Model output is None in epoch {epoch+1}, batch. Skipping.")
            continue
        
        loss = criterion(outputs, masks)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print(f"Warning: NaN loss detected in epoch {epoch+1}, batch. Skipping update.")
            # Consider logging inputs/outputs here for debugging
            continue
            
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), grad_clip)

        optimizer.step()
        
        epoch_loss += loss.item()
        # Detach outputs for metrics calculation
        batch_metrics = calculate_metrics(outputs.detach(), masks)
        epoch_metrics['IoU'] += batch_metrics['IoU']
        epoch_metrics['Dice'] += batch_metrics['Dice']
        
        train_progress.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dice': f"{batch_metrics['Dice']:.4f}"
        })
    
    avg_loss = epoch_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
    return avg_loss, avg_metrics

def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    epoch_metrics = {'IoU': 0.0, 'Dice': 0.0}
    num_batches = len(val_loader)
    val_progress = tqdm(val_loader, desc="[Val]", leave=False)
    
    with torch.no_grad():
        for batch in val_progress:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs, _ = model(images)
            
            if outputs is None:
                 print("Warning: Model output is None during validation, skipping batch.")
                 continue
                 
            loss = criterion(outputs, masks)
            
            epoch_loss += loss.item()
            batch_metrics = calculate_metrics(outputs, masks)
            epoch_metrics['IoU'] += batch_metrics['IoU']
            epoch_metrics['Dice'] += batch_metrics['Dice']
            
            val_progress.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{batch_metrics['Dice']:.4f}"
            })
            
    avg_loss = epoch_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
    return avg_loss, avg_metrics

def train_fold(args, fold_idx=None, num_folds=None, cross_val_indices=None):
    """Train model on a specific fold (or standard split if fold_idx is None)"""
    # Setup directories
    if fold_idx is not None:
        fold_dir = os.path.join(args.output_dir, f"fold_{fold_idx+1}")
        save_dir = Path(fold_dir)
    else:
        save_dir = Path(args.output_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = save_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # Save args
    with open(save_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
        fold_idx=fold_idx,
        num_folds=num_folds,
        cross_val_indices=cross_val_indices[fold_idx] if cross_val_indices else None
    )
    
    # Load base official model
    model_type = "vit_t"
    base_model = sam_model_registry[model_type](checkpoint=args.pretrained)
    print(f"Loaded official MobileSAM model (type: {model_type}) from {args.pretrained}")
    base_model = base_model.to(device)

    # Apply LoRA using the adapted wrapper
    # Ensure use_temp_head is True if not fine-tuning original decoder
    use_temp_head = not args.train_decoder
    lora_model = MobileSAM_LoRA_Adapted(
        model=base_model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        train_encoder=args.train_encoder,
        train_decoder=args.train_decoder,
        use_temp_head=use_temp_head
    ).to(device)

    criterion = CombinedLoss(dice_weight=args.dice_weight, focal_weight=args.focal_weight)
    
    # Get trainable parameters
    if args.train_only_lora:
        print("Configuring optimizer for LoRA parameters and segmentation head...")
        trainable_params = lora_model.get_trainable_parameters() # Gets LoRA + Head params
    else:
        print("Configuring optimizer for all trainable parameters...")
        trainable_params = filter(lambda p: p.requires_grad, lora_model.parameters())
        # Print count
        lora_model.get_trainable_parameters()

    # Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(trainable_params, lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    
    # Scheduler
    scheduler = None
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    elif args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_gamma, patience=args.lr_patience, verbose=True)
    
    best_val_loss = float('inf')
    best_val_dice = 0.0
    best_epoch = 0
    num_epochs_no_improvement = 0
    
    if fold_idx is not None:
        print(f"\nStarting training for Fold {fold_idx+1}/{num_folds}, {args.epochs} epochs...")
    else:
        print(f"\nStarting training for {args.epochs} epochs...")
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        train_loss, train_metrics = train_one_epoch(lora_model, train_loader, criterion, optimizer, device, epoch, args.epochs, args.grad_clip)
        val_loss, val_metrics = validate_one_epoch(lora_model, val_loader, criterion, device)
        
        epoch_duration = time.time() - epoch_start_time
        
        # Scheduler step logic
        if scheduler:
            if args.lr_scheduler == 'reduce':
                scheduler.step(val_metrics['Dice'])
            else:
                scheduler.step()
        
        # Logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('IoU/train', train_metrics['IoU'], epoch)
        writer.add_scalar('IoU/val', val_metrics['IoU'], epoch)
        writer.add_scalar('Dice/train', train_metrics['Dice'], epoch)
        writer.add_scalar('Dice/val', val_metrics['Dice'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Time/epoch', epoch_duration, epoch)
        
        fold_str = f"Fold {fold_idx+1}/{num_folds} " if fold_idx is not None else ""
        print(f"{fold_str}Epoch {epoch+1}/{args.epochs} [{epoch_duration:.2f}s] - Train Loss: {train_loss:.4f}, Train Dice: {train_metrics['Dice']:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_metrics['Dice']:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Checkpointing and Early Stopping
        if val_metrics['Dice'] > best_val_dice:
            best_val_dice = val_metrics['Dice']
            best_val_loss = val_loss
            best_epoch = epoch
            num_epochs_no_improvement = 0
            
            # Save the best model state dictionary
            torch.save({
                'epoch': epoch,
                'model_state_dict': lora_model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_dice': val_metrics['Dice'],
                'val_iou': val_metrics['IoU'],
                'args': vars(args),
                'fold': fold_idx
            }, save_dir / 'best_model.pth')
            
            print(f"---> Saved best model (Epoch {epoch+1}) with Val Dice: {best_val_dice:.4f}")
        else:
            num_epochs_no_improvement += 1
            print(f"---> No improvement in Val Dice for {num_epochs_no_improvement} epochs.")
            
        # Save periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': lora_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_dice': val_metrics['Dice'],
                'val_iou': val_metrics['IoU'],
                'args': vars(args),
                'fold': fold_idx
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
            print(f"---> Saved checkpoint at epoch {epoch+1}")
        
        # Early stopping check
        if args.early_stopping and num_epochs_no_improvement >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs without improvement.")
            break
    
    total_training_time = time.time() - start_time
    print(f"\nTotal Training Time: {total_training_time:.2f}s")
    
    # Final test evaluation using the best saved model
    print("\nLoading best model for final test evaluation...")
    best_model_path = save_dir / 'best_model.pth'
    test_metrics = {}
    
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        # Recreate model structure based on saved args
        saved_args = argparse.Namespace(**checkpoint['args']) 
        base_model_test = sam_model_registry[model_type](checkpoint=None).to(device) # Load architecture
        lora_model_test = MobileSAM_LoRA_Adapted(
             model=base_model_test,
             r=saved_args.lora_rank, 
             lora_alpha=saved_args.lora_alpha,
             lora_dropout=0.0, # No dropout for eval
             train_encoder=saved_args.train_encoder,
             train_decoder=saved_args.train_decoder,
             use_temp_head=not saved_args.train_decoder
        ).to(device)
        lora_model_test.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
        
        test_loss, test_metrics = validate_one_epoch(lora_model_test, test_loader, criterion, device)
        
        fold_str = f"Fold {fold_idx+1}/{num_folds} " if fold_idx is not None else ""
        print(f"\nFinal Test Results ({fold_str}using best model from epoch {checkpoint['epoch']+1}):")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test IoU:  {test_metrics['IoU']:.4f}")
        print(f"  Test Dice: {test_metrics['Dice']:.4f}")
        
        results_data = {
            'test_loss': test_loss,
            'test_iou': test_metrics['IoU'],
            'test_dice': test_metrics['Dice'],
            'best_epoch': best_epoch + 1, # 1-based index
            'best_val_loss': best_val_loss,
            'best_val_dice': best_val_dice,
            'fold': fold_idx,
            'total_training_time_seconds': total_training_time
        }
    else:
        print("Error: Best model checkpoint not found. Skipping test evaluation.")
        results_data = {"error": "Best model checkpoint not found"}

    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump(results_data, f, indent=4)
    
    writer.close()
    
    fold_str = f"Fold {fold_idx+1}/{num_folds} " if fold_idx is not None else ""
    print(f"\nTraining completed for {fold_str}. Best model saved at epoch {best_epoch+1} with Val Dice: {best_val_dice:.4f}")
    print(f"Results and logs saved in: {save_dir}")
    
    return best_val_dice, test_metrics.get('Dice', 0.0)

def train(args):
    """Main training function - handles k-fold or standard training"""
    if args.cross_validation:
        print(f"Running {args.num_folds}-fold cross-validation...")
        # Create cross-validation indices to ensure consistent folds
        cross_val_indices = create_cross_val_indices(args.data_dir, args.num_folds, args.seed)
        
        # Track fold metrics
        fold_metrics = {}
        
        for fold_idx in range(args.num_folds):
            val_dice, test_dice = train_fold(args, fold_idx, args.num_folds, cross_val_indices)
            fold_metrics[fold_idx] = {
                'val_dice': val_dice,
                'test_dice': test_dice
            }
        
        # Calculate average metrics
        avg_val_dice = np.mean([metrics['val_dice'] for metrics in fold_metrics.values()])
        avg_test_dice = np.mean([metrics['test_dice'] for metrics in fold_metrics.values()])
        
        # Save cross-validation summary
        summary = {
            'fold_metrics': fold_metrics,
            'avg_val_dice': avg_val_dice,
            'avg_test_dice': avg_test_dice,
            'args': vars(args)
        }
        
        with open(os.path.join(args.output_dir, 'cross_validation_results.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        
        print("\nCross-Validation Results Summary:")
        for fold_idx, metrics in fold_metrics.items():
            print(f"  Fold {fold_idx+1}/{args.num_folds}: Val Dice = {metrics['val_dice']:.4f}, Test Dice = {metrics['test_dice']:.4f}")
        print(f"  Average: Val Dice = {avg_val_dice:.4f}, Test Dice = {avg_test_dice:.4f}")
        
        return avg_val_dice, avg_test_dice
    else:
        # Standard training
        return train_fold(args)

def main_train(args):
    """Main training entry point"""
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    print("--- Training Configuration ---")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("-----------------------------")
    
    train(args)

# Updated arg parser with new parameters
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MobileSAM with LoRA on NuInsSeg dataset')
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='NuInsSeg', help='Path to NuInsSeg dataset')
    parser.add_argument('--image_size', type=int, default=1024, help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    # Model parameters
    parser.add_argument('--pretrained', type=str, default='weights/mobile_sam.pt', help='Path to pre-trained MobileSAM model')
    parser.add_argument('--train_encoder', action=argparse.BooleanOptionalAction, default=True, help='Train the image encoder via LoRA')
    parser.add_argument('--train_decoder', action=argparse.BooleanOptionalAction, default=False, help='Train the mask decoder via LoRA')
    parser.add_argument('--train_only_lora', action=argparse.BooleanOptionalAction, default=True, help='Train only LoRA parameters and segmentation head')
    # LoRA parameters
    parser.add_argument('--lora_rank', type=int, default=4, help='Rank of LoRA adaptation')
    parser.add_argument('--lora_alpha', type=int, default=4, help='Alpha scaling factor for LoRA')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='Dropout probability for LoRA layers')
    # Cross-validation parameters
    parser.add_argument('--cross_validation', action=argparse.BooleanOptionalAction, default=True, help='Use cross-validation')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for cross-validation')
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate for Cosine scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='Optimizer')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'step', 'reduce', 'none'], help='LR scheduler')
    parser.add_argument('--lr_step_size', type=int, default=30, help='Step size for StepLR scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Gamma for StepLR/ReduceLROnPlateau scheduler')
    parser.add_argument('--lr_patience', type=int, default=10, help='Patience for ReduceLROnPlateau scheduler (used with Val Dice)')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value (0 to disable)')
    # Loss parameters
    parser.add_argument('--dice_weight', type=float, default=0.5, help='Weight for Dice loss')
    parser.add_argument('--focal_weight', type=float, default=0.5, help='Weight for Focal loss')
    # Other parameters
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save checkpoints and logs')
    parser.add_argument('--save_interval', type=int, default=10, help='Epochs between checkpoint saves')
    parser.add_argument('--early_stopping', action=argparse.BooleanOptionalAction, default=True, help='Enable early stopping based on Val Dice')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main_train(args) 