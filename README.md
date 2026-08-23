# Flooded Road and Building Mapping from Satellite Imagery using Deep Learning

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![SpaceNet 8](https://img.shields.io/badge/Dataset-SpaceNet%208-blue.svg)](https://spacenet.ai/sn8-challenge/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)



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
│
├── api/                              # FastAPI Backend Server
│   ├── __init__.py
│   ├── main.py                       # Application entrypoint, CORS, static file serving
│   └── routes.py                     # /api/predict, /api/route, /api/sample, /api/health endpoints
│
├── checkpoints/                      # Saved Model Weights & Metrics
│   ├── .gitkeep
│   ├── final_eval_benchmark.json     # Comparative evaluation results (Fusion vs U-Net)
│   ├── fusion_metrics.csv            # Per-epoch training metrics for Siamese Fusion model
│   └── unet_metrics.csv             # Per-epoch training metrics for U-Net baseline
│   # NOTE: .pth weights excluded from git (>500MB). Train via Colab or download from Releases.
│
├── data/                             # Data Pipeline & Preprocessing
│   ├── __init__.py
│   ├── crs.py                        # Coordinate Reference System normalization (EPSG reprojection)
│   ├── dataset.py                    # PyTorch Dataset, manifest builder, GeoTIFF loader, train/val/test splits
│   ├── mask_generator.py             # 5-channel binary mask rasterizer from GeoJSON & CSV annotations
│   └── tiling.py                     # Satellite image tiling with configurable overlap
│
├── graph/                            # Road Network Topology & Routing Engine
│   ├── __init__.py
│   └── road_graph.py                 # Skeletonization → NetworkX graph → Flood attribution → Dijkstra routing
│
├── inference/                        # Model Inference Pipeline
│   ├── __init__.py
│   ├── eval.py                       # Per-class IoU evaluation with global confusion matrix
│   └── predict.py                    # FloodPredictor: sliding-window tiled inference with overlap stitching
│
├── losses/                           # Loss Functions
│   ├── __init__.py
│   └── combined_loss.py              # Multi-task loss: Dice + Focal + Cross-Entropy with class weighting
│
├── models/                           # Neural Network Architectures
│   ├── __init__.py
│   ├── siamese_fusion.py             # O3: Dual-stream Siamese encoder + Cross-Attention + FPN decoder
│   └── unet_baseline.py              # O2: Standard ResNet34 U-Net (6-channel early concatenation)
│
├── scripts/                          # Utility Scripts
│   ├── generate_colab_notebook.py    # Auto-generates the Colab training notebook
│   ├── setup_colab.sh                # One-click Colab environment setup (dataset download + preprocessing)
│   └── tile_images.py                # CLI tool for GeoTIFF tiling and mask generation
│
├── tests/                            # Automated Test Suite
│   ├── __init__.py
│   ├── conftest.py                   # Pytest fixtures and shared test configuration
│   ├── test_api.py                   # API endpoint integration tests
│   ├── test_data.py                  # Dataset loading and mask generation tests
│   ├── test_models.py                # Model forward pass shape validation tests
│   └── test_predict_stub.py          # Stub-mode prediction pipeline tests
│
├── ui/                               # React Frontend Dashboard
│   ├── index.html                    # HTML entry point
│   ├── package.json                  # NPM dependencies (React, Axios, Leaflet)
│   ├── vite.config.js                # Vite dev server config (port 3000, API proxy → 8000)
│   ├── public/
│   │   └── .gitkeep
│   └── src/
│       ├── main.jsx                  # React DOM mount point
│       ├── App.jsx                   # Main application: canvas rendering, tabs, routing UI
│       └── index.css                 # Swiss Minimalist White design system
│
├── notebooks/
│   └── SpaceNet8_Colab_Training.ipynb  # Interactive Colab notebook for cloud GPU training
│
├── colab1-train-logs.md              # Training logs: initial run (pre-fix metrics)
├── colab2-train-logs.md              # Training logs: second run (architecture debugging)
├── colab3.1-train-logs.md            # Training logs: final run (50 epochs Fusion + 25 epochs U-Net)
│
├── train.py                          # Training loop: AMP, OneCycleLR, global confusion matrix mIoU
├── eval.py                           # Comparative model evaluation and benchmark generation
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # CUDA-enabled container definition
├── docker-compose.yml                # Multi-container service orchestrator
├── .env                              # Environment variables
├── .gitignore                        # Git exclusions (datasets, weights, node_modules)
└── README.md                         # Project documentation (this file)
```

---

### 0. Training on Google Colab (T4 GPU)

You can train both models in Google Colab with a free **NVIDIA Tesla T4 GPU**:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isuryaprakashh/Capstone-Flood-Mapping/blob/main/notebooks/SpaceNet8_Colab_Training.ipynb)

Open [`notebooks/SpaceNet8_Colab_Training.ipynb`](notebooks/SpaceNet8_Colab_Training.ipynb) for step-by-step interactive training, automated Google Drive checkpoint backup, evaluation benchmarks, and visual qualitative inspections.

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

### 3. Model Training (Google Colab / Local GPU)

```bash
# Train the Novel Siamese Cross-Attention Fusion Network (O3)
python train.py --model fusion --data-root ./CP1-DATASET --epochs 40 --batch-size 8 --amp

# Train the Baseline U-Net (O2)
python train.py --model unet --data-root ./CP1-DATASET --epochs 25 --batch-size 8 --amp
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
