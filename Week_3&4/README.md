# Brain Tumor Segmentation (BraTS 2021) with Optimized U-Net Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)


## Model Download Links
https://drive.google.com/drive/folders/1y5O3yvnsUsvdwL2I4EqSkE-HzH7mA_DV?usp=sharing

## Kaggle Notebook Descriptions
Here is the link to the Kaggle Notebook which contains all my work :- https://www.kaggle.com/code/ndstab/braintumoursegmentation

### There are 10 versions in the notebook (Versions 1, 2, 3 and 4 are not relevant)
- Version 5 contains first full training run with some visulations and an average dice score of 0.8019.
- Version 6 contains a MultiDice Loss function along with training for more epochs (avg. dice score of 0.8267)
- Version 7 is just prediction (with TTA) (score = 0.8332)
- Version 8 fine tunes the previous model with a weighted dice loss function (gives more weight to the Necrotic class)
- Version 9 is inferencing using the fine tuned model (score = 0.8334)
- Version 10 contains 15 different visualisations

## Project Overview
This project implements a high-performance deep learning pipeline for **3D Brain Tumor Segmentation** using the BraTS 2021 dataset. The objective is to segment MRI scans into three distinct tumor sub-regions: **Necrotic Tumor Core**, **Peritumoral Edema**, and **Enhancing Tumor**.

The solution features a custom **U-Net** architecture trained with an optimized pipeline that reduced training latency by **76%** and achieved a Dice Score of **>0.83**, utilizing advanced techniques like Weighted Fine-Tuning and Test-Time Augmentation.

## Key Technical Achievements
* **Performance Engineering:** Reduced training time from **50 mins/epoch to 12 mins/epoch (76% reduction)** by implementing **Automatic Mixed Precision (AMP)** and asynchronous data pre-fetching (`num_workers`, `pin_memory`).
* **Advanced Regularization:** Implemented **On-the-Fly Augmentations** (Albumentations) to prevent overfitting on the 15k slice dataset.
* **Class Imbalance Handling:** Engineered a **Weighted Dice Loss** function to heavily penalize errors on the difficult "Necrotic Core" class (Weight=4.0) during a targeted fine-tuning phase.
* **Inference Optimization:** Deployed **Test-Time Augmentation (TTA)**, ensembling predictions from multiple views (Flip/Rotate) to boost final accuracy without retraining.

## Final Results
Evaluated on **15,891** unseen validation slices using the best fine-tuned model with TTA:

| Class | Dice Score | IoU |
| :--- | :--- | :--- |
| **Best Average (Tumor Only)** | **0.8334** | **--** |
| Necrotic Core (Class 1) | 0.7765 | 0.7003 |
| Edema (Class 2) | 0.8547 | 0.7646 |
| Enhancing Tumor (Class 3) | 0.8691 | 0.7957 |

## Dataset & Preprocessing
The dataset consists of multi-modal MRI Scans (T1, T1ce, T2, FLAIR).
* **Input:** 4-Channel MRI Slices (128x128 or 240x240).
* **Output:** 4-Class Segmentation Mask.
    * 0: Background
    * 1: Necrotic Tumor Core
    * 2: Peritumoral Edema
    * 3: Enhancing Tumor
* **Preprocessing:** Slices were extracted from 3D volumes and saved as `.npy` files for efficient I/O.

## Methodology & Tech Stack

### 1. Architecture
* **Model:** U-Net (Encoder-Decoder architecture with Skip Connections).
* **Input Channels:** 4 (Multi-modal MRI).
* **Output Channels:** 4 (Softmax probabilities).

### 2. Training Pipeline
* **Loss Function:** Custom **Weighted Dice Loss** (Weights: `[0.1, 4.0, 1.0, 1.0]`).
* **Optimizer:** Adam (`lr=1e-4` -> `5e-5`) with **ReduceLROnPlateau** scheduler.
* **Hardware Acceleration:** NVIDIA T4 x2 (DataParallel) + Tensor Cores (FP16 AMP).

### 3. Optimization Strategy
| Optimization | Impact |
| :--- | :--- |
| **Mixed Precision (AMP)** | Reduced VRAM usage by ~40%, enabled larger batches. |
| **Pinned Memory** | Accelerated CPU-to-GPU data transfer. |
| **Test-Time Augmentation** | Improved robustness to edge cases during inference. |


## Installation & Usage

### Prerequisites
```bash
pip install torch torchvision albumentations tqdm matplotlib numpy