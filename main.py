# Name: Zachary Wood
# Date: 4/10/2025
# Assignment 3
# Course: CAP 5516

# Title: Parameter Efficient Fine-tuning Foundation Model for Nuclei Instance Segmentation

# Description: 
# 1. Dataset
# NuInsSeg: A Fully Annotated Dataset for Nuclei Instance Segmentation in H&E-Stained Histological Images [1].
# The dataset and detailed step-by-step instructions to generate related segmentation masks are publicly available at
# https://www.kaggle.com/datasets/ipateam/nuinsseg and https://github.com/masih4/NuInsSeg, respectively.
# The dataset can also be downloaded from here: https://zenodo.org/records/10518968

# This dataset is located in ./NuInsSeg/
# It contains subfolders such as: human bladder, human brain, etc.
# Each subfolder contains more folders such as distance maps, Imagj_zips, label masks, label masks modify, etc.


# 2. Task
# Review the dataset paper [1] and its corresponding GitHub repository [2] to familiarize yourself with the
# experimental setup. You will follow the experiment setting as described in the paper.
# For segmentation, we will utilize the Segment Anything Model (SAM) [3]. To enhance resource efficiency, select
# one of the optimized versions of the original SAM model: MobileSAM [4], EfficientSAM [5], or TinySAM [6]. You
# may choose any of these variants along with their pre-trained models. I highly suggest you use these efficient SAM
# models. However, if you have the resource to run the original SAM model, that is also fine.
# Assuming the use of the MobileSAM model, we will apply LoRA [7] for parameter-efficient fine-tuning to perform
# nuclei instance segmentation on the NuInsSeg dataset [1].


# 3. What to report
# 1) The details of your implementation of applying LoRA for efficient fine-tuning.
# 2) Follow Table 3 in the dataset paper [1] and report the average results based on five-fold cross-validation in
# terms of those metrics (e.g., Dice, AJI, PQ). I hope your results can be much better than those baseline
# methods in the paper.
# 3) Since we will use LoRA for fine tuning, report the # of tunable parameters.
# 4) Provide a few examples of visual comparison of the predicted segmentation masks and the ground truth
# masks.


# 4. What to submit
# (1) A report for this assignment. Specifically, a detailed network architecture framework (figure) is required. For
# example, how the LoRA layers are applied in your efficient fine-tuning framework. The implementation details are
# important.
# (2) Clean code and clear instructions (e.g., a readme file) to reproduce your results. If you choose to host the code on
# GitHub, please provide the GitHub link.


# Useful resources (e.g. examples of applying LoRA for SAM fine tuning)
# 1. Finetune SAM on your customized medical imaging dataset https://github.com/mazurowski-lab/finetune-
# SAM
# 2. Medical SAM Adapter https://github.com/SuperMedIntel/Medical-SAM-Adapter
# 3. https://github.com/tianrun-chen/SAM-Adapter-PyTorch
# 4. MeLo: Low-rank Adaptation is Better than Finetuning for Medical Image
# https://github.com/JamesQFreeman/LoRA-ViT
# 5. SAMed: https://github.com/hitachinsk/SAMed
# 6. How to build the best medical image segmentation algorithm using foundation models: a comprehensive
# empirical study with Segment Anything Model https://arxiv.org/pdf/2404.09957
# References
# [1] Mahbod, Amirreza, Christine Polak, Katharina Feldmann, Rumsha Khan, Katharina Gelles, Georg Dorffner,
# Ramona Woitek, Sepideh Hatamikia, and Isabella Ellinger. "NuInsSeg: A fully annotated dataset for nuclei instance
# segmentation in H&E-stained histological images." Scientific Data 11, no. 1 (2024): 295.
# https://arxiv.org/pdf/2308.01760
# [2] https://github.com/masih4/NuInsSeg?tab=readme-ov-file#codes-to-generate-segmentation-masks
# [3] Kirillov, Alexander, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao et al.
# "Segment anything." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4015-
# 4026. 2023.
# [4] Zhang, Chaoning, Dongshen Han, Yu Qiao, Jung Uk Kim, Sung-Ho Bae, Seungkyu Lee, and Choong Seon
# Hong. "Faster segment anything: Towards lightweight sam for mobile applications." arXiv preprint
# arXiv:2306.14289 (2023). [Code] https://github.com/ChaoningZhang/MobileSAM
# [5] Xiong, Yunyang, Bala Varadarajan, Lemeng Wu, Xiaoyu Xiang, Fanyi Xiao, Chenchen Zhu, Xiaoliang Dai et
# al. "Efficientsam: Leveraged masked image pretraining for efficient segment anything." In Proceedings of the
# IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16111-16121. 2024. [Code]
# https://github.com/yformer/EfficientSAM (You can choose the smallest model EfficientSAM-Tiny)
# [6] Shu, Han, Wenshuo Li, Yehui Tang, Yiman Zhang, Yihao Chen, Houqiang Li, Yunhe Wang, and Xinghao Chen.
# "Tinysam: Pushing the envelope for efficient segment anything model." arXiv preprint arXiv:2312.13789 (2023).
# [Code] https://github.com/xinghaochen/TinySAM
# [7] Hu, Edward J., Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and
# Weizhu Chen. "Lora: Low-rank adaptation of large language models." arXiv preprint arXiv:2106.09685 (2021).

import argparse
import sys
import torch
import numpy as np
import random
import warnings

# Filter specific warnings
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated")
warnings.filterwarnings("ignore", message="Importing from timm.models.registry is deprecated")
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_*")

# Import functions from other modules
from train import main_train
from evaluate import main_evaluate
from demo import main as main_demo # Renamed to avoid conflict
from download_weights import main as main_download

def set_seed(seed):
    """
    Establishes deterministic behavior for reproducible experiments.
    
    Sets random seeds for PyTorch, NumPy, and Python's random module to ensure
    consistent results across multiple runs with the same seed value.
    
    Args:
        seed: Integer seed value for random number generators
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Uncomment for complete determinism (may impact performance):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='Main script for MobileSAM Nuclei Segmentation')
    parser.add_argument('mode', choices=['train', 'evaluate', 'demo', 'download', 'parameter_search'], 
                        help='Operation mode: train, evaluate, demo, download weights, or parameter_search')
    parser.add_argument('--seed', type=int, default=42, help='Global random seed')
    
    # Add arguments common to multiple modes or allow unknown args
    # Alternatively, parse known args first and pass remaining to sub-scripts

    # Initial parse to get the mode
    args, remaining_argv = parser.parse_known_args()
    
    # Add mode-specific arguments
    if args.mode == 'train':
        # Add training-specific arguments
        train_parser = argparse.ArgumentParser(description='Training Arguments')
        train_parser.add_argument('--data_dir', type=str, default='NuInsSeg')
        train_parser.add_argument('--image_size', type=int, default=1024)
        train_parser.add_argument('--num_workers', type=int, default=4)
        train_parser.add_argument('--pretrained', type=str, default='weights/mobile_sam.pt')
        train_parser.add_argument('--train_encoder', action=argparse.BooleanOptionalAction, default=True)
        train_parser.add_argument('--train_decoder', action=argparse.BooleanOptionalAction, default=False)
        train_parser.add_argument('--train_only_lora', action=argparse.BooleanOptionalAction, default=True)
        train_parser.add_argument('--lora_rank', type=int, default=4)
        train_parser.add_argument('--lora_alpha', type=int, default=4)
        train_parser.add_argument('--lora_dropout', type=float, default=0.1)
        train_parser.add_argument('--batch_size', type=int, default=4)
        train_parser.add_argument('--epochs', type=int, default=100)
        train_parser.add_argument('--learning_rate', type=float, default=1e-4)
        train_parser.add_argument('--min_lr', type=float, default=1e-6)
        train_parser.add_argument('--weight_decay', type=float, default=1e-4)
        train_parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'])
        train_parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'step', 'reduce', 'none'])
        train_parser.add_argument('--lr_step_size', type=int, default=30)
        train_parser.add_argument('--lr_gamma', type=float, default=0.1)
        train_parser.add_argument('--lr_patience', type=int, default=10)
        train_parser.add_argument('--grad_clip', type=float, default=1.0)
        train_parser.add_argument('--dice_weight', type=float, default=0.5)
        train_parser.add_argument('--focal_weight', type=float, default=0.5)
        train_parser.add_argument('--output_dir', type=str, default='output')
        train_parser.add_argument('--save_interval', type=int, default=10)
        train_parser.add_argument('--early_stopping', action=argparse.BooleanOptionalAction, default=True)
        train_parser.add_argument('--patience', type=int, default=20)
        # Cross-validation parameters
        train_parser.add_argument('--cross_validation', action=argparse.BooleanOptionalAction, default=True)
        train_parser.add_argument('--num_folds', type=int, default=5)
        # Add seed again for consistency, it will be overwritten by the main parser's seed
        train_parser.add_argument('--seed', type=int, default=42)
        # Parse the *remaining* arguments using the train parser
        mode_args = train_parser.parse_args(remaining_argv)
        
    elif args.mode == 'evaluate':
        eval_parser = argparse.ArgumentParser(description='Evaluation Arguments')
        eval_parser.add_argument('--data_dir', type=str, default='NuInsSeg')
        eval_parser.add_argument('--image_size', type=int, default=1024)
        eval_parser.add_argument('--num_workers', type=int, default=4)
        eval_parser.add_argument('--model_path', type=str, required=True)
        eval_parser.add_argument('--batch_size', type=int, default=4)
        eval_parser.add_argument('--threshold', type=float, default=0.5)
        eval_parser.add_argument('--output_dir', type=str, default='evaluation_results')
        eval_parser.add_argument('--visualize', action=argparse.BooleanOptionalAction, default=True)
        eval_parser.add_argument('--num_visualizations', type=int, default=20)
        eval_parser.add_argument('--plot', action=argparse.BooleanOptionalAction, default=True)
        eval_parser.add_argument('--seed', type=int, default=42)
        mode_args = eval_parser.parse_args(remaining_argv)
        
    elif args.mode == 'demo':
        demo_parser = argparse.ArgumentParser(description='Demo Arguments')
        demo_parser.add_argument('--model_path', type=str, required=True)
        demo_parser.add_argument('--data_dir', type=str, default='NuInsSeg')
        demo_parser.add_argument('--image_size', type=int, default=1024)
        demo_parser.add_argument('--num_samples', type=int, default=5)
        demo_parser.add_argument('--threshold', type=float, default=0.5)
        demo_parser.add_argument('--output_dir', type=str, default='demo_outputs')
        demo_parser.add_argument('--cpu', action='store_true')
        mode_args = demo_parser.parse_args(remaining_argv)

    elif args.mode == 'download':
        dl_parser = argparse.ArgumentParser(description='Download Arguments')
        dl_parser.add_argument('--model', type=str, default="mobile_sam", choices=["mobile_sam", "all"])
        dl_parser.add_argument('--output_dir', type=str, default="weights")
        dl_parser.add_argument('--force', action="store_true")
        mode_args = dl_parser.parse_args(remaining_argv)
        
    elif args.mode == 'parameter_search':
        # Parameter search mode
        ps_parser = argparse.ArgumentParser(description='Parameter Search Arguments')
        ps_parser.add_argument('--data_dir', type=str, default='NuInsSeg')
        ps_parser.add_argument('--image_size', type=int, default=1024)
        ps_parser.add_argument('--num_workers', type=int, default=4)
        ps_parser.add_argument('--pretrained', type=str, default='weights/mobile_sam.pt')
        ps_parser.add_argument('--train_encoder', action=argparse.BooleanOptionalAction, default=True)
        ps_parser.add_argument('--train_decoder', action=argparse.BooleanOptionalAction, default=False)
        ps_parser.add_argument('--train_only_lora', action=argparse.BooleanOptionalAction, default=True)
        ps_parser.add_argument('--lora_dropout', type=float, default=0.1, help='Dropout probability for LoRA layers')
        ps_parser.add_argument('--batch_size', type=int, default=4)
        ps_parser.add_argument('--epochs', type=int, default=30)  # Shorter epochs for search
        ps_parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'])
        ps_parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'step', 'reduce', 'none'])
        ps_parser.add_argument('--lr_step_size', type=int, default=30)
        ps_parser.add_argument('--lr_gamma', type=float, default=0.1)
        ps_parser.add_argument('--lr_patience', type=int, default=10)
        ps_parser.add_argument('--min_lr', type=float, default=1e-6)
        ps_parser.add_argument('--weight_decay', type=float, default=1e-4)
        ps_parser.add_argument('--grad_clip', type=float, default=1.0)
        ps_parser.add_argument('--dice_weight', type=float, default=0.5)
        ps_parser.add_argument('--focal_weight', type=float, default=0.5)
        ps_parser.add_argument('--early_stopping', action=argparse.BooleanOptionalAction, default=True)
        ps_parser.add_argument('--patience', type=int, default=15)
        ps_parser.add_argument('--output_dir', type=str, default='parameter_search')
        ps_parser.add_argument('--save_interval', type=int, default=10)
        # Parameter search specific arguments
        ps_parser.add_argument('--lora_ranks', type=str, default='2,4,8,16', 
                               help='Comma-separated list of LoRA ranks to try')
        ps_parser.add_argument('--lora_alphas', type=str, default='2,4,8', 
                               help='Comma-separated list of LoRA alphas to try')
        ps_parser.add_argument('--learning_rates', type=str, default='1e-4,3e-4,1e-3', 
                               help='Comma-separated list of learning rates to try')
        # Cross-validation for parameter search
        ps_parser.add_argument('--num_folds', type=int, default=3)  # Smaller folds for faster search
        ps_parser.add_argument('--seed', type=int, default=42)
        mode_args = ps_parser.parse_args(remaining_argv)
        
    else:
        # Should not happen due to choices constraint
        parser.print_help()
        sys.exit(1)
        
    # Combine the main args (mode, seed) with the mode-specific args
    # Overwrite mode_args.seed with the global seed
    final_args = mode_args
    final_args.mode = args.mode 
    final_args.seed = args.seed 
    
    return final_args

def parameter_search(args):
    """
    Performs hyperparameter optimization for the LoRA architecture.
    
    This function conducts a grid search over specified hyperparameter combinations,
    specifically exploring different LoRA ranks, alpha scaling factors, and learning rates.
    For each configuration, it performs k-fold cross-validation training and tracks
    validation and test performance to identify optimal settings.
    
    Args:
        args: Namespace containing configuration parameters
        
    Returns:
        Dictionary containing the best hyperparameter configuration
    """
    # Parse parameter search space from comma-separated strings
    lora_ranks = [int(r) for r in args.lora_ranks.split(',')]
    lora_alphas = [int(a) for a in args.lora_alphas.split(',')]
    learning_rates = [float(lr) for lr in args.learning_rates.split(',')]
    
    print(f"Starting parameter search with:")
    print(f"  LoRA ranks: {lora_ranks}")
    print(f"  LoRA alphas: {lora_alphas}")
    print(f"  Learning rates: {learning_rates}")
    print(f"  Using {args.num_folds}-fold cross-validation for each configuration\n")
    
    # Create output directory for search results
    import os
    from pathlib import Path
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tracking variables
    best_dice = 0.0
    best_config = None
    results = {}
    
    # Setup for grid search
    from copy import deepcopy
    total_configs = len(lora_ranks) * len(lora_alphas) * len(learning_rates)
    config_idx = 0
    
    # Grid search across all hyperparameter combinations
    for lora_rank in lora_ranks:
        for lora_alpha in lora_alphas:
            for lr in learning_rates:
                config_idx += 1
                print(f"\n--- Configuration {config_idx}/{total_configs} ---")
                print(f"LoRA rank: {lora_rank}, Alpha: {lora_alpha}, LR: {lr}")
                
                # Create configuration-specific args
                config_args = deepcopy(args)
                config_args.lora_rank = lora_rank
                config_args.lora_alpha = lora_alpha
                config_args.learning_rate = lr
                config_args.cross_validation = True
                config_args.output_dir = os.path.join(args.output_dir, f"rank{lora_rank}_alpha{lora_alpha}_lr{lr}")
                
                # Execute training with cross-validation for this configuration
                from train import train
                val_dice, test_dice = train(config_args)
                
                # Record results for this configuration
                config_key = f"rank{lora_rank}_alpha{lora_alpha}_lr{lr}"
                results[config_key] = {
                    "val_dice": val_dice,
                    "test_dice": test_dice,
                    "lora_rank": lora_rank,
                    "lora_alpha": lora_alpha,
                    "learning_rate": lr
                }
                
                # Update best configuration if current one is superior
                if val_dice > best_dice:
                    best_dice = val_dice
                    best_config = {
                        "lora_rank": lora_rank,
                        "lora_alpha": lora_alpha,
                        "learning_rate": lr,
                        "val_dice": val_dice,
                        "test_dice": test_dice
                    }
    
    # Serialize and save results
    import json
    summary = {
        "results": results,
        "best_config": best_config
    }
    
    results_path = os.path.join(args.output_dir, "parameter_search_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=4)
    
    # Display sorted results summary
    print("\n=== Parameter Search Results ===")
    for config_key, result in sorted(results.items(), key=lambda x: x[1]["val_dice"], reverse=True):
        print(f"{config_key}: Val Dice = {result['val_dice']:.4f}, Test Dice = {result['test_dice']:.4f}")
    
    print(f"\nBest configuration: Rank {best_config['lora_rank']}, Alpha {best_config['lora_alpha']}, LR {best_config['learning_rate']}")
    print(f"Best validation Dice: {best_config['val_dice']:.4f}")
    print(f"Best test Dice: {best_config['test_dice']:.4f}")
    print(f"Results saved to {results_path}")
    
    return best_config

if __name__ == "__main__":
    args = parse_args()
    
    # Set the global random seed
    print(f"Setting global random seed to: {args.seed}")
    set_seed(args.seed)
    
    # Execute the selected mode
    if args.mode == 'train':
        print("\n--- Running Training ---")
        main_train(args)
    elif args.mode == 'evaluate':
        print("\n--- Running Evaluation ---")
        main_evaluate(args)
    elif args.mode == 'demo':
        print("\n--- Running Demo ---")
        main_demo(args)
    elif args.mode == 'download':
        print("\n--- Running Download ---")
        main_download(args)
    elif args.mode == 'parameter_search':
        print("\n--- Running Parameter Search ---")
        parameter_search(args)
        
    print(f"\n--- {args.mode.capitalize()} finished ---")



# Download pre-trained MobileSAM weights:
# python main.py download --output_dir weights

# Parameter search to find optimal LoRA configuration:
# python main.py parameter_search --data_dir NuInsSeg --output_dir parameter_search --lora_ranks 2,4,8,16 --lora_alphas 2,4,8 --learning_rates 1e-4,3e-4,1e-3 --epochs 30 --num_folds 3

# Train the model with optimal configuration and 5-fold cross-validation:
# python main.py train --data_dir NuInsSeg --output_dir output_lora_optimal --pretrained weights/mobile_sam.pt --batch_size 4 --learning_rate 3e-4 --epochs 50 --early_stopping --patience 15 --lora_rank 8 --lora_alpha 8 --train_only_lora --cross_validation --num_folds 5

# Standard training without cross-validation (if preferred):
# python main.py train --data_dir NuInsSeg --output_dir output_lora_standard --pretrained weights/mobile_sam.pt --batch_size 4 --learning_rate 3e-4 --epochs 50 --early_stopping --patience 15 --lora_rank 8 --lora_alpha 8 --train_only_lora --no-cross_validation

# Evaluate model performance (repeat for each fold as needed):
# python main.py evaluate --data_dir NuInsSeg --model_path output_lora_optimal/fold_1/best_model.pth --output_dir evaluation_lora_optimal/fold_1 --visualize --plot

# Run the demo to visualize results on sample images:
# python main.py demo --data_dir NuInsSeg --model_path output_lora_optimal/fold_1/best_model.pth --output_dir demo_lora_optimal --num_samples 10

# Evaluation Results from fold 1 (Full results in the Readme):
# Metrics by tissue type:
#   human bladder (3 samples): IoU: 0.7313, Dice: 0.8447, AJI: 0.5787, PQ: 0.3991
#   human cardia (4 samples): IoU: 0.7238, Dice: 0.8381, AJI: 0.4811, PQ: 0.2309
#   human cerebellum (2 samples): IoU: 0.7960, Dice: 0.8864, AJI: 0.6458, PQ: 0.4582
#   human epiglottis (2 samples): IoU: 0.7785, Dice: 0.8716, AJI: 0.7043, PQ: 0.6031
#   human jejunum (2 samples): IoU: 0.7585, Dice: 0.8627, AJI: 0.2417, PQ: 0.1815
#   human kidney (2 samples): IoU: 0.8181, Dice: 0.9000, AJI: 0.4343, PQ: 0.2590
#   human liver (12 samples): IoU: 0.7297, Dice: 0.8431, AJI: 0.5937, PQ: 0.4325
#   human lung (1 samples): IoU: 0.7622, Dice: 0.8650, AJI: 0.6000, PQ: 0.4016
#   human melanoma (2 samples): IoU: 0.5194, Dice: 0.6836, AJI: 0.4211, PQ: 0.2469
#   human muscle (1 samples): IoU: 0.4345, Dice: 0.6058, AJI: 0.6678, PQ: 0.3562
#   human oesophagus (9 samples): IoU: 0.7936, Dice: 0.8842, AJI: 0.6474, PQ: 0.5501
#   human pancreas (12 samples): IoU: 0.7250, Dice: 0.8399, AJI: 0.5721, PQ: 0.4075
#   human peritoneum (2 samples): IoU: 0.6199, Dice: 0.7631, AJI: 0.5567, PQ: 0.3671
#   human placenta (9 samples): IoU: 0.8135, Dice: 0.8953, AJI: 0.4982, PQ: 0.3546
#   human pylorus (2 samples): IoU: 0.7786, Dice: 0.8752, AJI: 0.6298, PQ: 0.4271
#   human salivory gland (6 samples): IoU: 0.7015, Dice: 0.8228, AJI: 0.4531, PQ: 0.3206
#   human spleen (10 samples): IoU: 0.7333, Dice: 0.8453, AJI: 0.4377, PQ: 0.2706
#   human testis (2 samples): IoU: 0.6673, Dice: 0.8004, AJI: 0.6581, PQ: 0.5379
#   human tongue (5 samples): IoU: 0.7260, Dice: 0.8351, AJI: 0.5379, PQ: 0.4196
#   human tonsile (1 samples): IoU: 0.6507, Dice: 0.7884, AJI: 0.5427, PQ: 0.2501
#   human umbilical cord (3 samples): IoU: 0.4620, Dice: 0.6314, AJI: 0.4499, PQ: 0.1748
#   mouse fat (white and brown)_subscapula (5 samples): IoU: 0.4777, Dice: 0.6374, AJI: 0.5714, PQ: 0.3955
#   mouse femur (2 samples): IoU: 0.5562, Dice: 0.7128, AJI: 0.5038, PQ: 0.2183
#   mouse heart (3 samples): IoU: 0.4855, Dice: 0.6517, AJI: 0.4825, PQ: 0.2436
#   mouse kidney (8 samples): IoU: 0.6153, Dice: 0.7589, AJI: 0.5897, PQ: 0.4561
#   mouse liver (9 samples): IoU: 0.4350, Dice: 0.5948, AJI: 0.5454, PQ: 0.3568
#   mouse muscle_tibia (7 samples): IoU: 0.5287, Dice: 0.6802, AJI: 0.5450, PQ: 0.4123
#   mouse spleen (6 samples): IoU: 0.6974, Dice: 0.8210, AJI: 0.1901, PQ: 0.0853
#   mouse thymus (2 samples): IoU: 0.7388, Dice: 0.8498, AJI: 0.1277, PQ: 0.0767

# Overall Results:
#   IoU:  0.6726
#   Dice: 0.7952
#   AJI:  0.5197
#   PQ:   0.3606
#   SQ:   0.7384
#   RQ:   0.4851
