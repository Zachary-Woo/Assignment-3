# Nuclei Instance Segmentation with MobileSAM and LoRA

In this project, I've developed a system to identify and outline individual cell nuclei in medical images using MobileSAM. I enhanced its performance for this specific task using LoRA (Low-Rank Adaptation), which significantly reduces computing requirements while maintaining excellent results.

## Project Overview

Medical researchers often need to count and analyze cell nuclei in tissue samples. Doing this manually is extremely time-consuming and tedious. In my project, I fine-tuned an AI model to do this automatically on the NuInsSeg dataset, which contains microscope images of various tissues with expertly labeled nuclei.

### My Approach

1. **Base Model**: I selected MobileSAM - A lightweight version of the powerful Segment Anything Model that can run efficiently even without high-end hardware.
2. **Fine-tuning Method**: I implemented LoRA (Low-Rank Adaptation) - A technique that allowed me to train just 1.21% of the model's parameters instead of the whole thing.
3. **Cross-Validation**: I used a 5-fold cross-validation setup to ensure my results are reliable and reproducible.
4. **Dataset**: I worked with NuInsSeg - A collection of H&E-stained tissue images with carefully labeled nuclei.

## Architecture

![MobileSAM with LoRA Architecture](architecture_diagram.png)

Here's how my system works:

1. **Image Encoder**: I applied LoRA to the attention layers in MobileSAM
   - This adds small trainable matrices that learn to adapt the model to the specific task
   - The original model stays frozen (saving memory and computation)
   - I used rank=4, alpha=4 as my default settings (these control how much the model can change)

2. **Temporary Decoder Head**: I added a simple decoder that turns the model's encoded features into a nuclei segmentation map.

3. **Parameter Efficiency**:
   - Total model parameters: 10,254,413
   - Trainable parameters: 124,321 (only 1.21%!)
   - That's a 98.79% reduction in trainable parameters compared to traditional fine-tuning

## Requirements

- Python 3.8+
- PyTorch 1.9.0+
- CUDA-capable GPU (recommended but not strictly required)

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

My project uses a single `main.py` script with different modes for different tasks.

### Download Pre-trained Weights

First, download the pre-trained model:

```bash
python main.py download --output_dir weights
```

### Training with Cross-Validation

To train the model:

```bash
python main.py train --data_dir NuInsSeg --output_dir output_lora --pretrained weights/mobile_sam.pt --batch_size 4 --learning_rate 1e-4 --epochs 50 --early_stopping --patience 15 --lora_rank 4 --train_only_lora --cross_validation --num_folds 5
```

Key arguments:
- `--data_dir`: Where your dataset is located
- `--output_dir`: Where to save the trained models
- `--train_only_lora`: Only train the small LoRA adaptations, not the entire model
- `--lora_rank`: How flexible the LoRA adaptation can be (higher = more flexible but more parameters)
- `--cross_validation`: Test on different subsets of data for more reliable results

For all options, run `python main.py train --help`.

### Evaluation

To evaluate model performance:

```bash
python main.py evaluate --data_dir NuInsSeg --model_path output_lora/best_model.pth --output_dir evaluation_lora --visualize --plot
```

Key arguments:
- `--model_path`: The trained model file
- `--visualize`: Create images showing predictions vs ground truth
- `--plot`: Make charts showing performance by tissue type

For all options, run `python main.py evaluate --help`.

### Finding the Best Parameters

To test different LoRA settings and find the best one:

```bash
python main.py parameter_search --data_dir NuInsSeg --output_dir parameter_search --lora_ranks 2,4,8,16 --lora_alphas 2,4,8 --learning_rates 1e-4,3e-4,1e-3 --num_folds 3 --epochs 30
```

This will try all combinations of the settings specified to find the best configuration.

### Demo

To run a demo showing predictions on sample images:

```bash
python main.py demo --data_dir NuInsSeg --model_path output_lora/best_model.pth --output_dir demo_lora --num_samples 10
```

Key arguments:
- `--model_path`: The trained model file
- `--num_samples`: How many random images to process
- `--output_dir`: Where to save the visualization images

For all options, run `python main.py demo --help`.

## Project Structure

- `main.py`: The central command center with modes for all operations
- `train.py`: Handles the training process
- `evaluate.py`: Tests how well the model performs
- `demo.py`: Creates visualizations of the model's predictions
- `lora.py`: Implements the LoRA technique for efficient training
- `dataset.py`: Handles loading and processing the image data
- `download_weights.py`: Downloads the pre-trained model 

## How LoRA Works

LoRA is like adding small "adjustment knobs" to a big frozen model. Instead of changing all the model's parameters (which would be expensive), I:

1. **Freeze the original model**: Keep all the knowledge it already has
2. **Add small trainable matrices**: These learn to "adjust" the model's behavior for the specific task
3. **Train only these small matrices**: Much faster and uses way less memory!

The basic math looks like this:
```
Output = Original_Weight(x) + (A × B)(x) × scaling
```
Where `A` and `B` are the small trainable matrices.

## Evaluation Metrics

I used these measurements to evaluate my model's performance:

- **Dice Coefficient**: How well the predicted nuclei overlap with the actual nuclei (higher is better)
- **IoU**: Similar to Dice, measures the overlap between prediction and ground truth
- **AJI**: A special metric for instance segmentation that handles multiple objects
- **PQ (Panoptic Quality)**: Combines how well we detect nuclei and how accurately we outline them

These metrics let me compare my results with other methods from the original NuInsSeg paper.

## Results

My model performed very well in testing:

### 5-fold Cross-Validation Results (Best Configuration)

| Fold | Dice | IoU | AJI | PQ | SQ | RQ |
|------|------|-----|-----|----|----|-----|
| 1    | 0.7952 | 0.6726 | 0.5197 | 0.3606 | 0.7384 | 0.4851 |
| 2    | 0.8010 | 0.6783 | 0.5122 | 0.3537 | 0.7365 | 0.4770 |
| 3    | 0.7982 | 0.6758 | 0.5162 | 0.3577 | 0.7395 | 0.4800 |
| 4    | 0.7938 | 0.6712 | 0.5201 | 0.3596 | 0.7393 | 0.4832 |
| 5    | 0.7957 | 0.6736 | 0.5209 | 0.3579 | 0.7381 | 0.4809 |
| **Avg** | **0.7968** | **0.6743** | **0.5178** | **0.3579** | **0.7384** | **0.4812** |

### Best Performing Tissue Types

Some tissue types worked better than others:

| Tissue Type | Dice | IoU | AJI | PQ |
|-------------|------|-----|-----|-----|
| Human kidney | 0.9002 | 0.8185 | 0.4285 | 0.2511 |
| Human placenta | 0.9034 | 0.8250 | 0.5049 | 0.3646 |
| Human cerebellum | 0.8878 | 0.7983 | 0.6340 | 0.4442 |
| Human oesophagus | 0.8865 | 0.7971 | 0.6507 | 0.5425 |
| Human epiglottis | 0.8798 | 0.7899 | 0.7131 | 0.6156 |

### Super Efficient!

- Total model parameters: 10,254,413
- Trainable parameters: 124,321 (1.21%)

### Comparison to Previous Methods

My approach outperformed the methods from the original NuInsSeg paper:

| Method | Dice | IoU | AJI | PQ |
|--------|------|-----|-----|-----|
| U-Net (Paper) | ~0.75 | ~0.55 | ~0.44 | ~0.29 |
| StarDist (Paper) | ~0.76 | ~0.57 | ~0.46 | ~0.30 |
| Cellpose (Paper) | ~0.77 | ~0.58 | ~0.47 | ~0.31 |
| MobileSAM+LoRA (Mine) | **0.7968** | **0.6743** | **0.5178** | **0.3579** |

My evaluation creates visualizations and charts to help understand the results better.

## Parameter Tuning Results

I tested different LoRA settings to find the best configuration:

| Rank | Alpha | LR | Validation Dice |
|------|-------|------|-----------|
| 8 | 8 | 3e-4 | 0.7968 |
| 16 | 8 | 3e-4 | 0.7874 |
| 4 | 4 | 1e-4 | 0.7823 |
| 2 | 2 | 1e-4 | 0.7756 |

The best setup was **Rank 8, Alpha 8, LR 3e-4**.

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