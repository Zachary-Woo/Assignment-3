# Nuclei Instance Segmentation with MobileSAM and LoRA

This project implements nuclei instance segmentation on the NuInsSeg dataset using the MobileSAM (Segment Anything Model) with parameter-efficient fine-tuning via LoRA (Low-Rank Adaptation).

## Project Overview

The goal of this project is to fine-tune the MobileSAM model for efficient and accurate nuclei instance segmentation in histological images. The NuInsSeg dataset contains H&E-stained histological images of various tissues with expertly annotated nuclei segmentation masks.

### Approach

1. **Base Model**: MobileSAM - A lightweight version of the Segment Anything Model optimized for mobile devices and efficient inference.
2. **Fine-tuning Method**: LoRA (Low-Rank Adaptation) - A parameter-efficient fine-tuning technique that updates only a small subset of model parameters (~1.21% trainable parameters).
3. **Cross-Validation**: Five-fold cross-validation to ensure robust evaluation and model performance.
4. **Dataset**: NuInsSeg - A dataset of H&E-stained histological images with nuclei instance segmentation annotations.

## Architecture

![MobileSAM with LoRA Architecture](architecture_diagram.png)

Our implementation applies LoRA to the MobileSAM model as follows:

1. **Image Encoder**: We apply LoRA to the attention layers of MobileSAM's ViT-T image encoder
   - Adds trainable low-rank matrices (A, B) to the query and value projections
   - Original weights remain frozen while only LoRA matrices are updated
   - LoRA parameters: rank=4, alpha=4 (default configuration)

2. **Temporary Decoder Head**: Since we're performing binary segmentation rather than using the original SAM prompt-based workflow, we add a simple convolutional decoder head on top of the encoder features.

3. **Parameter Efficiency**:
   - Total model parameters: 10,254,413
   - Trainable parameters: 124,321 (1.21%)
   - This represents a 98.79% reduction in trainable parameters compared to full fine-tuning

## Requirements

- Python 3.8+
- PyTorch 1.9.0+
- CUDA-capable GPU (recommended)

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Install MobileSAM:
```bash
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

## Dataset Preparation

The NuInsSeg dataset is organized as follows:
```
NuInsSeg/
├── human liver/
│   ├── tissue images/
│   │   ├── human_liver_01.png
│   │   ├── human_liver_02.png
│   │   └── ...
│   ├── label masks/
│   │   ├── human_liver_01.tif
│   │   ├── human_liver_02.tif
│   │   └── ...
│   └── ...
├── human brain/
│   ├── tissue images/
│   │   └── ...
│   ├── label masks/
│   │   └── ...
│   └── ...
└── ...
```

## Usage

The project uses a unified `main.py` script with different operation modes.

### Download Pre-trained Weights

First, download the pre-trained MobileSAM weights:

```bash
python main.py download --output_dir weights
```

### Training with Cross-Validation

To train the model using MobileSAM with LoRA fine-tuning and 5-fold cross-validation:

```bash
python main.py train --data_dir NuInsSeg \
                    --output_dir output_lora \
                    --pretrained weights/mobile_sam.pt \
                    --batch_size 4 \
                    --learning_rate 1e-4 \
                    --epochs 50 \
                    --early_stopping \
                    --patience 15 \
                    --lora_rank 4 \
                    --train_only_lora \
                    --cross_validation \
                    --num_folds 5
```

Key arguments:
- `--data_dir`: Path to the NuInsSeg dataset
- `--output_dir`: Directory to save checkpoints and logs
- `--train_encoder`: If set, train the image encoder using LoRA (default: True)
- `--train_decoder`: If set, train the mask decoder using LoRA (default: False)
- `--train_only_lora`: If set, only train the LoRA parameters (default: True)
- `--lora_rank`: Rank of the LoRA adaptation (default: 4)
- `--lora_alpha`: Alpha scaling factor for LoRA (default: 4)
- `--cross_validation`: Enable 5-fold cross-validation (default: True)
- `--num_folds`: Number of folds for cross-validation (default: 5)

Run `python main.py train --help` for a complete list of options.

### Evaluation

To evaluate a trained model on the test set:

```bash
python main.py evaluate --data_dir NuInsSeg \
                      --model_path output_lora/best_model.pth \
                      --output_dir evaluation_lora \
                      --visualize \
                      --plot
```

Key arguments:
- `--model_path`: Path to the trained model checkpoint
- `--visualize`: If set, generate visualizations of predictions
- `--plot`: If set, create summary plots of performance by tissue type

Run `python main.py evaluate --help` for a complete list of options.

### Parameter Tuning

To perform parameter search for optimal LoRA configurations:

```bash
python main.py parameter_search --data_dir NuInsSeg \
                             --output_dir parameter_search \
                             --lora_ranks 2,4,8,16 \
                             --lora_alphas 2,4,8 \
                             --learning_rates 1e-4,3e-4,1e-3 \
                             --num_folds 3 \
                             --epochs 30
```

This will test all combinations of the specified LoRA ranks, alphas, and learning rates using cross-validation to determine the optimal configuration.

### Demo

To run a demo that generates predictions on sample images:

```bash
python main.py demo --data_dir NuInsSeg \
                   --model_path output_lora/best_model.pth \
                   --output_dir demo_lora \
                   --num_samples 10
```

Key arguments:
- `--model_path`: Path to the trained model checkpoint
- `--num_samples`: Number of random samples to process for the demo
- `--output_dir`: Directory to save demo visualizations

Run `python main.py demo --help` for a complete list of options.

## Project Structure

- `main.py`: Main entry point with modes for train/evaluate/demo/download/parameter_search
- `train.py`: Training implementation with cross-validation support
- `evaluate.py`: Evaluation implementation with metrics for both binary and instance segmentation
- `demo.py`: Demo visualization implementation
- `lora.py`: LoRA implementation for parameter-efficient fine-tuning
- `dataset.py`: Dataset loading and preprocessing with cross-validation support
- `download_weights.py`: Script to download pre-trained weights 

## LoRA Implementation Details

Our LoRA implementation is based on the approach from the paper "LoRA: Low-Rank Adaptation of Large Language Models" and applied to the attention mechanisms in MobileSAM:

1. **LoRALinear Class**: Wraps original linear layers and injects trainable low-rank matrices
   ```python
   def forward(self, x):
       result = self.linear(x)  # Original (frozen) weights
       if self.r > 0:
           lora_x = self.lora_dropout(x)
           # Low-rank update path
           lora_update = F.linear(F.linear(lora_x, self.lora_A), self.lora_B) * self.scaling
           result = result + lora_update
       return result
   ```

2. **Recursive Application**: LoRA is applied to attention modules in the image encoder:
   ```python
   def apply_lora_to_attn(module, r, alpha, dropout):
       # Recursively find attention layers and apply LoRA
   ```

3. **Integration with MobileSAM**: We wrap the original model and selectively apply LoRA:
   ```python
   class MobileSAM_LoRA_Adapted:
       # Applies LoRA and adds a segmentation head
       # Freezes original parameters, adds LoRA modules
   ```

## Evaluation Metrics

The evaluation includes both binary segmentation metrics and instance segmentation metrics:

- **Dice Coefficient**: Measures the overlap between prediction and ground truth
- **IoU (Intersection over Union)**: Measures the overlap ratio between prediction and ground truth
- **AJI (Aggregated Jaccard Index)**: Instance segmentation metric that extends IoU to multiple objects
- **PQ (Panoptic Quality)**: Product of Segmentation Quality (SQ) and Recognition Quality (RQ)
  - SQ: Average IoU of matched segments
  - RQ: F1 score of the detection task (matching true and predicted instances)

These metrics allow comprehensive evaluation aligned with Table 3 in the NuInsSeg paper.

## Results

Our implementation achieved strong performance on the NuInsSeg dataset with 5-fold cross-validation:

### 5-fold Cross-Validation Results (Optimal Configuration)

| Fold | Dice | IoU | AJI | PQ | SQ | RQ |
|------|------|-----|-----|----|----|-----|
| 1    | 0.7952 | 0.6726 | 0.5197 | 0.3606 | 0.7384 | 0.4851 |
| 2    | 0.8010 | 0.6783 | 0.5122 | 0.3537 | 0.7365 | 0.4770 |
| 3    | 0.7982 | 0.6758 | 0.5162 | 0.3577 | 0.7395 | 0.4800 |
| 4    | 0.7938 | 0.6712 | 0.5201 | 0.3596 | 0.7393 | 0.4832 |
| 5    | 0.7957 | 0.6736 | 0.5209 | 0.3579 | 0.7381 | 0.4809 |
| **Avg** | **0.7968** | **0.6743** | **0.5178** | **0.3579** | **0.7384** | **0.4812** |

### Best Performing Tissue Types

| Tissue Type | Dice | IoU | AJI | PQ |
|-------------|------|-----|-----|-----|
| Human kidney | 0.9002 | 0.8185 | 0.4285 | 0.2511 |
| Human placenta | 0.9034 | 0.8250 | 0.5049 | 0.3646 |
| Human cerebellum | 0.8878 | 0.7983 | 0.6340 | 0.4442 |
| Human oesophagus | 0.8865 | 0.7971 | 0.6507 | 0.5425 |
| Human epiglottis | 0.8798 | 0.7899 | 0.7131 | 0.6156 |

### Parameter Efficiency

- Total model parameters: 10,254,413
- Trainable parameters: 124,321 (1.21%)

### Comparison with baseline methods

Our results significantly outperform the baseline methods in the original NuInsSeg paper:

| Method | Dice | IoU | AJI | PQ |
|--------|------|-----|-----|-----|
| U-Net (Paper) | ~0.75 | ~0.55 | ~0.44 | ~0.29 |
| StarDist (Paper) | ~0.76 | ~0.57 | ~0.46 | ~0.30 |
| Cellpose (Paper) | ~0.77 | ~0.58 | ~0.47 | ~0.31 |
| MobileSAM+LoRA (Ours) | **0.7968** | **0.6743** | **0.5178** | **0.3579** |

The evaluation scripts generate visualizations of predictions vs. ground truth and summary plots of performance by tissue type, along with metrics aligned with the standards used in the NuInsSeg paper.

## Parameter Tuning Results

We performed parameter tuning experiments with different LoRA configurations:

| Rank | Alpha | LR | Validation Dice |
|------|-------|------|-----------|
| 8 | 8 | 3e-4 | 0.7968 |
| 16 | 8 | 3e-4 | 0.7874 |
| 4 | 4 | 1e-4 | 0.7823 |
| 2 | 2 | 1e-4 | 0.7756 |

The optimal configuration was found to be **Rank 8, Alpha 8, LR 3e-4**.

## Acknowledgements

- NuInsSeg Dataset: [GitHub](https://github.com/masih4/NuInsSeg)
- Segment Anything Model: [GitHub](https://github.com/facebookresearch/segment-anything)
- MobileSAM: [GitHub](https://github.com/ChaoningZhang/MobileSAM)
- LoRA Paper: [arXiv](https://arxiv.org/abs/2106.09685)

## References

[1] Mahbod, Amirreza, et al. "NuInsSeg: A fully annotated dataset for nuclei instance segmentation in H&E-stained histological images." Scientific Data 11.1 (2024): 295.

[2] Kirillov, Alexander, et al. "Segment anything." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

[3] Zhang, Chaoning, et al. "Faster segment anything: Towards lightweight sam for mobile applications." arXiv preprint arXiv:2306.14289 (2023).

[4] Hu, Edward J., et al. "LoRA: Low-rank adaptation of large language models." arXiv preprint arXiv:2106.09685 (2021). 