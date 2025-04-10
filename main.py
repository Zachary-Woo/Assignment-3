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

