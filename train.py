import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
from tqdm import tqdm
import json
import time
from datetime import datetime
from pathlib import Path

from dataset import create_dataloaders
from mobile_sam import setup_model
from lora import MobileSAM_LoRA

# Loss functions
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        pred: (B, C, H, W) - predicted masks
        target: (B, H, W) - ground truth instance segmentation mask
        """
        # Convert instance segmentation to binary segmentation
        # If mask > 0, it's foreground
        binary_target = (target > 0).float()
        
        # Convert to one-hot if the pred is multi-class
        if pred.shape[1] > 1:
            pred = torch.sigmoid(pred)
            pred = pred[:, 0, :, :]  # Take only the first mask
        else:
            pred = torch.sigmoid(pred.squeeze(1))
        
        # Flatten the tensors
        pred_flat = pred.view(-1)
        target_flat = binary_target.view(-1)
        
        # Calculate Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        pred: (B, C, H, W) - predicted masks
        target: (B, H, W) - ground truth instance segmentation mask
        """
        # Convert instance segmentation to binary segmentation
        binary_target = (target > 0).float()
        
        # Convert to one-hot if the pred is multi-class
        if pred.shape[1] > 1:
            pred = torch.sigmoid(pred)
            pred = pred[:, 0, :, :]  # Take only the first mask
        else:
            pred = torch.sigmoid(pred.squeeze(1))
        
        # Focal loss computation
        binary_target = binary_target.view(-1)
        pred = pred.view(-1)
        
        # Apply sigmoid to get probabilities
        pt = pred * binary_target + (1 - pred) * (1 - binary_target)
        alpha_factor = self.alpha * binary_target + (1 - self.alpha) * (1 - binary_target)
        modulating_factor = (1.0 - pt) ** self.gamma
        
        loss = -alpha_factor * modulating_factor * torch.log(pt + 1e-8)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# Combined loss
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5, smooth=1.0, alpha=0.25, gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(smooth)
        self.focal_loss = FocalLoss(alpha, gamma)
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.dice_weight * dice + self.focal_weight * focal


# Evaluation metrics
def calculate_metrics(pred, target, threshold=0.5):
    """
    Calculate IoU and Dice score for binary segmentation
    
    Args:
        pred (torch.Tensor): Predicted mask (B, 1, H, W) or (B, H, W)
        target (torch.Tensor): Target mask (B, H, W)
        threshold (float): Threshold for binarizing predictions
    
    Returns:
        dict: Dictionary containing IoU and Dice scores
    """
    if pred.shape[1] > 1:
        pred = pred[:, 0, :, :]  # Take only the first mask
    else:
        pred = pred.squeeze(1)
    
    pred = (pred > threshold).float()
    binary_target = (target > 0).float()
    
    # Compute intersection and union
    intersection = (pred * binary_target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + binary_target.sum(dim=(1, 2)) - intersection
    
    # Calculate IoU
    iou = (intersection + 1e-8) / (union + 1e-8)
    mean_iou = iou.mean().item()
    
    # Calculate Dice score
    dice = (2 * intersection + 1e-8) / (pred.sum(dim=(1, 2)) + binary_target.sum(dim=(1, 2)) + 1e-8)
    mean_dice = dice.mean().item()
    
    return {
        'IoU': mean_iou,
        'Dice': mean_dice
    }


def train(args):
    """
    Main training function
    """
    # Create directory for saving checkpoints and logs
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    # Save training args
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    
    # Load pre-trained model
    if args.pretrained:
        model = setup_model(args.pretrained)
        print(f"Loaded pre-trained model from {args.pretrained}")
    else:
        model = setup_model()
        print("Using model without pre-trained weights")
    
    # Apply LoRA
    lora_model = MobileSAM_LoRA(
        model=model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        train_encoder=args.train_encoder,
        train_decoder=args.train_decoder
    )
    
    # Move model to device
    lora_model = lora_model.to(device)
    
    # Define loss function and optimizer
    criterion = CombinedLoss(
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight
    )
    
    # Get trainable parameters
    if args.train_only_lora:
        trainable_params = lora_model.get_trainable_parameters()
    else:
        trainable_params = lora_model.parameters()
    
    # Create optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(trainable_params, lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    elif args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_gamma, patience=args.lr_patience, verbose=True
        )
    
    # Track best validation metrics
    best_val_loss = float('inf')
    best_val_dice = 0.0
    best_epoch = 0
    num_epochs_no_improvement = 0
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # Training phase
        lora_model.train()
        train_loss = 0.0
        train_metrics = {'IoU': 0.0, 'Dice': 0.0}
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch_idx, batch in enumerate(train_progress):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = lora_model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            loss.backward()
            
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                
            optimizer.step()
            
            # Update train loss and metrics
            train_loss += loss.item()
            batch_metrics = calculate_metrics(outputs.detach(), masks)
            train_metrics['IoU'] += batch_metrics['IoU']
            train_metrics['Dice'] += batch_metrics['Dice']
            
            # Update progress bar
            train_progress.set_postfix({
                'loss': loss.item(),
                'dice': batch_metrics['Dice']
            })
        
        # Calculate average training metrics
        train_loss /= len(train_loader)
        train_metrics['IoU'] /= len(train_loader)
        train_metrics['Dice'] /= len(train_loader)
        
        # Validation phase
        lora_model.eval()
        val_loss = 0.0
        val_metrics = {'IoU': 0.0, 'Dice': 0.0}
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for batch_idx, batch in enumerate(val_progress):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                # Forward pass
                outputs, _ = lora_model(images)
                loss = criterion(outputs, masks)
                
                # Update validation loss and metrics
                val_loss += loss.item()
                batch_metrics = calculate_metrics(outputs, masks)
                val_metrics['IoU'] += batch_metrics['IoU']
                val_metrics['Dice'] += batch_metrics['Dice']
                
                # Update progress bar
                val_progress.set_postfix({
                    'loss': loss.item(),
                    'dice': batch_metrics['Dice']
                })
        
        # Calculate average validation metrics
        val_loss /= len(val_loader)
        val_metrics['IoU'] /= len(val_loader)
        val_metrics['Dice'] /= len(val_loader)
        
        # Update learning rate scheduler
        if args.lr_scheduler == 'reduce':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('IoU/train', train_metrics['IoU'], epoch)
        writer.add_scalar('IoU/val', val_metrics['IoU'], epoch)
        writer.add_scalar('Dice/train', train_metrics['Dice'], epoch)
        writer.add_scalar('Dice/val', val_metrics['Dice'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Dice: {train_metrics['Dice']:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Dice: {val_metrics['Dice']:.4f}")
        
        # Check if this is the best model based on validation Dice score
        if val_metrics['Dice'] > best_val_dice:
            best_val_dice = val_metrics['Dice']
            best_val_loss = val_loss
            best_epoch = epoch
            num_epochs_no_improvement = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': lora_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_dice': val_metrics['Dice'],
                'val_iou': val_metrics['IoU'],
                'args': vars(args)
            }, os.path.join(args.output_dir, 'best_model.pth'))
            
            print(f"Saved best model with Dice: {best_val_dice:.4f}")
        else:
            num_epochs_no_improvement += 1
            
        # Save checkpoint every save_interval epochs
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': lora_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_dice': val_metrics['Dice'],
                'val_iou': val_metrics['IoU'],
                'args': vars(args)
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Early stopping
        if args.early_stopping and num_epochs_no_improvement >= args.patience:
            print(f"Early stopping after {num_epochs_no_improvement} epochs without improvement.")
            break
    
    # Final evaluation on test set
    lora_model.eval()
    test_loss = 0.0
    test_metrics = {'IoU': 0.0, 'Dice': 0.0}
    
    with torch.no_grad():
        test_progress = tqdm(test_loader, desc="Testing")
        for batch_idx, batch in enumerate(test_progress):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs, _ = lora_model(images)
            loss = criterion(outputs, masks)
            
            # Update test loss and metrics
            test_loss += loss.item()
            batch_metrics = calculate_metrics(outputs, masks)
            test_metrics['IoU'] += batch_metrics['IoU']
            test_metrics['Dice'] += batch_metrics['Dice']
            
            # Update progress bar
            test_progress.set_postfix({
                'loss': loss.item(),
                'dice': batch_metrics['Dice']
            })
    
    # Calculate average test metrics
    test_loss /= len(test_loader)
    test_metrics['IoU'] /= len(test_loader)
    test_metrics['Dice'] /= len(test_loader)
    
    # Print final test results
    print(f"Test Loss: {test_loss:.4f}, "
          f"Test IoU: {test_metrics['IoU']:.4f}, "
          f"Test Dice: {test_metrics['Dice']:.4f}")
    
    # Log test metrics
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump({
            'test_loss': test_loss,
            'test_iou': test_metrics['IoU'],
            'test_dice': test_metrics['Dice'],
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'best_val_dice': best_val_dice
        }, f, indent=4)
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"Training completed. Best model saved at epoch {best_epoch+1} with Dice: {best_val_dice:.4f}")
    
    return best_val_dice, test_metrics['Dice']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MobileSAM with LoRA on NuInsSeg dataset')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='NuInsSeg', help='Path to NuInsSeg dataset')
    parser.add_argument('--image_size', type=int, default=1024, help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Model parameters
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pre-trained MobileSAM model')
    parser.add_argument('--train_encoder', action='store_true', help='Train the image encoder')
    parser.add_argument('--train_decoder', action='store_true', help='Train the mask decoder')
    parser.add_argument('--train_only_lora', action='store_true', help='Train only LoRA parameters')
    
    # LoRA parameters
    parser.add_argument('--lora_rank', type=int, default=4, help='Rank of LoRA adaptation')
    parser.add_argument('--lora_alpha', type=int, default=4, help='Alpha scaling factor for LoRA')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='Dropout probability for LoRA layers')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='Optimizer')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'step', 'reduce'], help='LR scheduler')
    parser.add_argument('--lr_step_size', type=int, default=30, help='Step size for StepLR scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Gamma for StepLR scheduler')
    parser.add_argument('--lr_patience', type=int, default=10, help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    
    # Loss parameters
    parser.add_argument('--dice_weight', type=float, default=0.5, help='Weight for Dice loss')
    parser.add_argument('--focal_weight', type=float, default=0.5, help='Weight for Focal loss')
    
    # Other parameters
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save checkpoints and logs')
    parser.add_argument('--save_interval', type=int, default=10, help='Epochs between checkpoint saves')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # Print arguments
    print("Training arguments:")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    
    # Start training
    train(args) 