# Nuclei Instance Segmentation with MobileSAM and LoRA

This repository contains code for performing nuclei instance segmentation on the NuInsSeg dataset using the MobileSAM (Segment Anything Model) with parameter-efficient fine-tuning via LoRA (Low-Rank Adaptation).

## Project Overview

The goal of this project is to fine-tune the MobileSAM model for efficient and accurate nuclei instance segmentation in histological images. The NuInsSeg dataset contains H&E-stained histological images of various tissues with expertly annotated nuclei segmentation masks.

### Approach

1. **Base Model**: MobileSAM - A lightweight version of the Segment Anything Model optimized for mobile devices and efficient inference.
2. **Fine-tuning Method**: LoRA (Low-Rank Adaptation) - A parameter-efficient fine-tuning technique that updates only a small subset of model parameters.
3. **Dataset**: NuInsSeg - A dataset of H&E-stained histological images with nuclei instance segmentation annotations.

## Requirements

- Python 3.8+
- PyTorch 1.9.0+
- CUDA-capable GPU (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nuclei-instance-segmentation.git
cd nuclei-instance-segmentation
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the MobileSAM pre-trained weights or use the model without pre-trained weights.

## Dataset Preparation

The NuInsSeg dataset should be organized as follows:
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

### Training

To train the model using MobileSAM with LoRA fine-tuning:

```bash
python train.py --data_dir NuInsSeg \
                --output_dir output \
                --train_encoder \
                --train_decoder \
                --train_only_lora \
                --lora_rank 4 \
                --lora_alpha 4 \
                --batch_size 4 \
                --learning_rate 1e-4 \
                --epochs 100 \
                --optimizer adamw \
                --early_stopping \
                --patience 20
```

Key arguments:
- `--data_dir`: Path to the NuInsSeg dataset
- `--output_dir`: Directory to save checkpoints and logs
- `--train_encoder`: If set, train the image encoder using LoRA
- `--train_decoder`: If set, train the mask decoder using LoRA
- `--train_only_lora`: If set, only train the LoRA parameters
- `--lora_rank`: Rank of the LoRA adaptation
- `--lora_alpha`: Alpha scaling factor for LoRA
- `--pretrained`: Path to pre-trained MobileSAM weights (optional)

Run `python train.py --help` for a complete list of options.

### Evaluation

To evaluate a trained model on the test set:

```bash
python evaluate.py --data_dir NuInsSeg \
                  --model_path output/best_model.pth \
                  --output_dir evaluation_results \
                  --visualize \
                  --plot
```

Key arguments:
- `--model_path`: Path to the trained model checkpoint
- `--visualize`: If set, generate visualizations of predictions
- `--plot`: If set, create summary plots of performance by tissue type

Run `python evaluate.py --help` for a complete list of options.

## Project Structure

- `dataset.py`: Dataset loading and preprocessing classes
- `mobile_sam.py`: MobileSAM model implementation
- `lora.py`: LoRA implementation for parameter-efficient fine-tuning
- `train.py`: Training script
- `evaluate.py`: Evaluation script
- `requirements.txt`: Required packages

## Results

After training, the model should be able to segment nuclei in histological images with high accuracy. The evaluation script will generate:

1. Overall IoU and Dice scores for the test set
2. Per-tissue type performance metrics
3. Visualizations of predictions vs. ground truth (if `--visualize` is set)
4. Summary plots of performance by tissue type (if `--plot` is set)

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