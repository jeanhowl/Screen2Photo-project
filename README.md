# Screen Degradation Synthesis (Camrip Simulation)

This repository contains a hybrid neural + procedural pipeline for generating synthetic “camrip-like” images from clean movie frames.


---

## Problem Statement

Given a clean frame (e.g. movie still), generate an image that mimics real-world screen capture artifacts such as:

* perspective distortion (camera angle)
* blur and edge smoothing
* sensor noise
* padding / borders
* color and tone shifts



---

## Approach

The system combines a learnable model with a procedural post-processing pipeline.

### Neural Component

The neural part is trained on paired data:

```
cover (clean image) → final (real photo of screen)
```

It consists of:

* **Generator**: HDRNet-style architecture
* **Alignment module**: Spatial Transformer Network (STN)
* **Discriminator**: Multi-scale conditional discriminator (PatchGAN-like)

### Generator (HDRNet)

* predicts local and global adjustments

* uses residual connection:

  ```
  output = input + adjustment
  ```

* operates in normalized image space [-1, 1]

### Spatial Transformer Network (STN)

* applied to target images during training
* learns affine transformation for alignment
* reduces geometric mismatch between generated and target images

### Discriminator

* multi-scale discriminator

* operates on concatenated pairs:

  ```
  [input, target] vs [input, generated]
  ```

* returns intermediate feature maps (used for feature matching loss)

---

## Loss Functions

The generator is trained with a combination of:

* **Adversarial loss (GAN)**
* **L1 reconstruction loss**
* **Feature Matching loss**

Total loss:

```
L_G = L_GAN + λ1 * L1 + λ_feat * L_feature_matching
```

Typical values:

```
λ1 = 2.0
λ_feat = 10.0
```

---

## Procedural Post-processing

After neural generation, additional artifacts are applied:

* perspective transform
* random rotation
* padding with dark borders
* Gaussian blur
* additive Gaussian noise

This step simulates real-world camera capture effects that are hard to learn purely from data.

---

## Metrics

Evaluation is performed using:

* MSE (Mean Squared Error)
* PSNR (Peak Signal-to-Noise Ratio)
* SSIM (Structural Similarity)

Note: these metrics do not fully reflect perceptual realism of degradation, but are used for consistency.

---

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── augmentations.py        # Procedural transformations (noise, perspective, etc.)
│   ├── dataset.py              # PairedImageDataset
│   ├── models.py               # HDRNet, STN, Discriminators
│   ├── metrics.py              # MSE, PSNR, SSIM
│   ├── checkpoint.py           # Save/load checkpoints
│   ├── train.py                # Training loop
│   └── infer.py                # Inference pipeline
│
├── data/
│   ├── train/                  # Training dataset (cover/final pairs)
│   └── inference/              # Inference dataset
│
├── outputs/
│   ├── checkpoints/            # Saved model checkpoints (.pth)
│   └── inference/              # Generated images and results
│
├── requirements.txt
├── .gitignore
└── README.md
```


---

## Installation

```
pip install -r requirements.txt
```

---

## Training

```
python train.py --config configs/train.yaml
```

Example config:

```
training:
  epochs: 200
  lr: 1e-4
  lambda_l1: 2.0
  lambda_feat: 10.0
```

---

## Inference

```
python infer.py \
    --checkpoint checkpoints/model.pth \
    --input data/samples/cover \
    --output outputs/examples
```

---

## Example Results

The repository includes sample outputs:

```
input → generated → postprocessed → target
```

(see `outputs/examples/`)

---

## Limitations

* full training dataset is not included
* Pretrained weights are not included due to size constraints. If you need to check it - contact me via @jeanhowl in Telegram.

---

## What This Project Demonstrates

* PyTorch-based image-to-image training pipeline
* custom Dataset and DataLoader
* manual data collecting for augmentation (initial dataset was not enough)
* adversarial training (GAN)
* feature matching loss
* spatial transformer integration
* reproducible inference pipeline
* hybrid neural + procedural modeling

---

## Tech Stack

* PyTorch
* torchvision
* OpenCV
* PIL
* NumPy
* scikit-image

---
