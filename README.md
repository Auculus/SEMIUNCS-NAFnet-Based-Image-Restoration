# AI-Based Restoration of Degraded Images

## Team

- **Pranav** — Model Development & Training
- **Shreyash** — Data Processing & Evaluation
- **Praket** — System Integration & Deployment

---

## 1. Project Overview

This project addresses the problem of **AI-based restoration of degraded semiconductor images**.

The system uses a deep learning-based image restoration and super-resolution model to recover high-quality images from degraded low-resolution inputs.

The model is designed to handle:

- Speckle noise
- Gaussian noise
- Low-resolution / super-resolution degradation

The final system takes a degraded image as input and produces a restored high-resolution image as output.

---

## 2. Model Architecture

The proposed solution is based on a modified **NAFNet V3** architecture implemented using PyTorch.

The network consists of:

- Input convolution
- Multi-level encoder
- NAF blocks
- Simplified Channel Attention (SCA)
- SimpleGate activation
- Learnable residual scaling
- Bottleneck / middle blocks
- Multi-level decoder
- Pixel Shuffle upsampling
- Residual reconstruction using bicubic interpolation

### Super-Resolution

The model performs **2× super-resolution**.

For an input image of size: H x W

The model produces: 2H x 2W

---

## 3. Repository Structure

```text
SEMICON/
│
├── Configs/
│   └── v2_updated.yaml
│
├── Models/
│   └── nafnet_v3.py
│
├── Data/
│   ├── train/
│   │   └── train/
│   │       ├── NoisyLR/
│   │       └── GT/
│   │
│   └── Test_NoisyLR/
│       └── NoisyLR/
│
├── checkpoints/
│   └── v3/
│       └── best_model.pth
│
├── restored_test_outputs/
│   └── ...
│
├── train_v3.py
│
├── evaluate.py
│
├── requirements.txt
│
└── README.md
```

## 4. Clone the Repository

For cloning the repository, in gitbash Run:
```text
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd SEMICON
```

Then setup Venv:
```text
python -m venv venv
venv\Scripts\activate
```


Then instal dependencies:
```text
pip install -r requirements.txt
```

NOTE: THE TRAINING FILE IS run.py AND THE FILE FOR TESTING AND EVALUATING THE MODEL IS visualize_v3.py
