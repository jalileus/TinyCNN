# TinyCNN Training for Embedded Face Detection

## Author: Alhajali Jalal

This repository contains the complete training pipeline for a **tiny CNN** designed to run on microcontrollers (STM32, etc.) as a **false positive filter for Haar cascade face detection**.

The trained model (34KB, int8 quantized) is meant to be used with the [`haar_cnn_demo`](https://github.com/Jalileus/opencv_samples) in Embox.

## Overview

This project provides a complete pipeline to train a tiny CNN that acts as a **verifier stage** after Haar cascade detection:

## Dataset

The provided `Dataset/` folder contains a pre-generated balanced dataset:

| Split      | Faces | Non-faces | Total | Source                 |
| ---------- | ----- | --------- | ----- | ---------------------- |
| Train      | 4,900 | 4,200     | 9,100 | LFW + PASS             |
| Validation | 1,050 | 900       | 1,950 | LFW + PASS             |
| Test       | 1,050 | 900       | 1,950 | LFW + PASS             |

**Image size:** 34×34

**Face source:** [LFW Dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset/data) (13,000+ images, person-aware splitting)

**Non-face source:** [PASS Dataset](https://www.robots.ox.ac.uk/~vgg/data/pass/) (150k images) — only Haar false positives are used as hard negatives

## Model Architecture: TinyCNN

```
Input: 34×34 grayscale (1,156 bytes)
    ↓
Conv1 (8 filters, 3×3, padding=1) → ReLU → MaxPool2d (2×2) → [17×17×8]
    ↓
Conv2 (16 filters, 3×3, padding=1) → ReLU → MaxPool2d (2×2) → [8×8×16]
    ↓
Flatten (1024) → FC1 (32) → ReLU → FC2 (1) → Sigmoid
```

**Total parameters:** 34,081

**Float32 size:** 136 KB

**Int8 quantized size:** 34 KB (with per-layer scaling)

## Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/yourusername/TinyCNN.git
cd TinyCNN
```

### 2. Install dependencies

```bash
pip install opencv-python torch numpy matplotlib scikit-learn tqdm jupyter
```

### 3. Explore the dataset

```bash
# Check dataset statistics
ls -la Dataset/train/faces/ | wc -l
ls -la Dataset/train/nonfaces/ | wc -l
```

### 4. Run the training notebook

```bash
jupyter notebook TinyCNN.ipynb
```

The notebook guides you through:
- Loading the dataset
- Defining the model architecture
- Training (20 epochs, ~2-5 minutes on CPU)
- Validation and testing
- ROC curve analysis
- Quantization and weight export

### 5. (Optional) Generate your own dataset

If you want to create a custom dataset:

check the jupyter notebook **Dataset_mining.ipynb**

This shows how to:
- Mine hard negatives from any non-face dataset
- Extract faces from LFW or your own face dataset
- Apply augmentations
- Create train/val/test splits

## Exporting for Embox Deployment

After training, the notebook automatically exports:

- **`stm32_weights/`** folder containing int8 quantized weights as C headers:
  - `conv1_weight.h`, `conv1_bias.h`
  - `conv2_weight.h`, `conv2_bias.h`
  - `fc1_weight.h`, `fc1_bias.h`
  - `fc2_weight.h`, `fc2_bias.h`
  - `cnn_model.h` (main header)

These files can be copied directly to the [`haar_cnn_demo`](https://github.com/jalileus/opencv_samples) in your Embox opencv_samples fork.

## requirements.txt

```
opencv-python==4.8.1.78
torch==2.1.0
numpy==1.24.3
matplotlib==3.7.2
scikit-learn==1.3.0
tqdm==4.65.0
jupyter==1.0.0
```

## How to Cite This Work

If you use this in your research, please cite:

```
@misc{TinyCNN2026,
  author = {Your Name},
  title = {TinyCNN Training for Embedded Face Detection},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/TinyCNN}
}
```

## Acknowledgments

- [LFW Dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset/data) - Face images
- [PASS Dataset](https://www.robots.ox.ac.uk/~vgg/data/pass/) - Hard negative mining
- Embox developers for the RTOS platform
- OpenCV community

## Contact

For questions or collaborations:

- Name: Alhazhali.zh@phystech.edu
- GitHub: [@jalileus](https://github.com/jalileus)
- Repository: [https://github.com/jalileus/TinyCNN](https://github.com/jalileus/TinyCNN)

---