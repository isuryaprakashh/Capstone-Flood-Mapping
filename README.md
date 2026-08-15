# Flooded Road and Building Mapping from Satellite Imagery using Deep Learning

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![SpaceNet 8](https://img.shields.io/badge/Dataset-SpaceNet%208-blue.svg)](https://spacenet.ai/sn8-challenge/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Capstone Project ID: `KLCAP-2026-00010`**

An end-to-end deep learning framework and rescue operations dashboard for automated detection of flooded roads and damaged buildings from multi-temporal (pre/post event) satellite imagery.

---

## 🌟 Key Highlights & Objectives

- **O1 (Data Pipeline)**: Automated GeoTIFF tiling, coordinate reference system (CRS) normalization, and 5-channel binary mask rasterization from SpaceNet 8 GeoJSON/CSV annotations.
- **O2 (Baseline Model)**: ResNet34-backed U-Net architecture processing early-concatenated pre/post temporal imagery.
- **O3 (Novel Siamese Cross-Attention Network)**: Dual-stream encoder architecture with bidirectional cross-attention mechanisms capturing subtle temporal changes and flood damage patterns.
- **O4 (Web Platform & Routing Engine)**: FastAPI asynchronous microservice and React/Leaflet interactive GIS map with NetworkX graph-based flood-avoiding emergency routing.
- **O5 (MLOps & Containerization)**: Production-ready testing suite (pytest), NVIDIA GPU Docker container, and experiment tracking with Weights & Biases.

---

## 📁 Codebase Architecture

```
CAPSTONE-PROJECT/
├── api/                  # FastAPI backend server & endpoints (/predict, /route)
├── checkpoints/          # Saved model weights (.pth)
├── CP1-DATASET/          # SpaceNet 8 dataset (Germany, Louisiana East/West)
├── data/                 # CRS normalization, GeoTIFF tiling, mask generator, PyTorch dataset
├── graph/                # NetworkX road network topology & flood-safe routing engine
├── inference/            # Prediction pipeline & sliding-window inference with overlap stitching
├── losses/               # Multi-task loss functions (Dice + Focal + BCE)
├── models/               # U-Net Baseline & Siamese Cross-Attention Fusion Network
├── scripts/              # Data preprocessing and manifest generation tools
├── tests/                # Comprehensive unit and integration test suite
├── ui/                   # React + Leaflet interactive frontend dashboard
├── Dockerfile            # CUDA-enabled container definition
├── docker-compose.yml    # Multi-container service orchestrator
├── train.py              # Mixed-precision training pipeline with OneCycleLR
└── eval.py               # Comparative model evaluation script
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/isuryaprakashh/Capstone-Flood-Mapping.git
cd Capstone-Flood-Mapping

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preprocessing & Mask Generation

```bash
# Generate 5-channel segmentation masks from GeoJSON & reference CSVs
python scripts/tile_images.py --data-root ./CP1-DATASET --output-dir ./processed
```

### 3. Model Training (HPC / GPU Server)

```bash
# Train the Baseline U-Net
python train.py --model unet --data-root ./CP1-DATASET --epochs 50 --batch-size 8

# Train the Novel Siamese Cross-Attention Fusion Network
python train.py --model fusion --data-root ./CP1-DATASET --epochs 100 --batch-size 8
```

### 4. Model Evaluation & Benchmark Comparison

```bash
python eval.py \
  --data-root ./CP1-DATASET \
  --unet-ckpt checkpoints/unet_net_best.pth \
  --fusion-ckpt checkpoints/fusion_net_best.pth
```

### 5. Running the Web System & API

```bash
# Start FastAPI backend (with stub mode fallback or trained weights)
python api/main.py

# In another terminal, run the React frontend
cd ui
npm install
npm run dev
```

Visit the interactive platform at `http://localhost:3000`.

---

## 🧪 Testing Suite

Run full automated tests:

```bash
python -m pytest tests/ -v
```

---

## 📊 Mask Channels Definition

| Channel Index | Target Feature | Description |
|---|---|---|
| `Channel 0` | `building` | Structural building footprint polygon |
| `Channel 1` | `road` | Road network centerline linestring |
| `Channel 2` | `flood` | Flood extent overlay |
| `Channel 3` | `flooded_building` | Submerged or damaged building footprint |
| `Channel 4` | `flooded_road` | Impassable flooded road segment |

---

## 👥 Authors

- **KL University Capstone Project KLCAP-2026-00010**
- Department of Computer Science & Engineering
