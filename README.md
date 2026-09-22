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
├── backend/                          # FastAPI Backend Server
│   ├── __init__.py
│   ├── main.py                       # Application entrypoint, CORS, static file serving
│   └── routes.py                     # /api/predict, /api/route, /api/sample, /api/health endpoints
│
├── frontend/                         # React Frontend Dashboard
│   ├── index.html                    # HTML entry point
│   ├── package.json                  # NPM dependencies (React, Axios, Canvas)
│   ├── vite.config.js                # Vite dev server config (port 3000, API proxy → 8000)
│   ├── public/
│   │   └── .gitkeep
│   └── src/
│       ├── main.jsx                  # React DOM mount point
│       ├── App.jsx                   # Main application: canvas rendering, tabs, routing UI
│       └── index.css                 # Swiss Minimalist White design system
│
├── dataloader/                       # Data Pipeline & Preprocessing
│   ├── __init__.py
│   ├── crs.py                        # Coordinate Reference System normalization (EPSG reprojection)
│   ├── dataset.py                    # PyTorch Dataset, manifest builder, GeoTIFF loader, train/val/test splits
│   ├── mask_generator.py             # 5-channel binary mask rasterizer from GeoJSON & CSV annotations
│   └── tiling.py                     # Satellite image tiling with configurable overlap
│
├── models/                           # Neural Network Architectures
│   ├── __init__.py
│   ├── siamese_fusion.py             # O3: Dual-stream Siamese encoder + Cross-Attention + FPN decoder
│   └── unet_baseline.py              # O2: Standard ResNet34 U-Net (6-channel early concatenation)
│
├── losses/                           # Loss Functions
│   ├── __init__.py
│   └── combined_loss.py              # Multi-task loss: Dice + Focal + Cross-Entropy with class weighting
│
├── routing/                          # Road Network Topology & Routing Engine
│   ├── __init__.py
│   └── road_graph.py                 # Skeletonization → NetworkX graph → Flood attribution → Dijkstra routing
│
├── inference/                        # Model Inference Pipeline
│   ├── __init__.py
│   ├── eval.py                       # Per-class IoU evaluation with global confusion matrix
│   └── predict.py                    # FloodPredictor: sliding-window tiled inference with overlap stitching
│
├── weights/                          # Model Weights & Metrics
│   ├── final_eval_benchmark.json     # Comparative evaluation results (Fusion vs U-Net)
│   ├── fusion_metrics.csv            # Per-epoch training metrics for Siamese Fusion model
│   └── unet_metrics.csv              # Per-epoch training metrics for U-Net baseline
│   # NOTE: .pth weights excluded from git (>500MB). Train via Colab or download from Releases.
│
├── scripts/                          # Utility Scripts
│   ├── eval.py                       # Comparative model evaluation and benchmark generation
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
├── notebooks/
│   └── SpaceNet8_Colab_Training.ipynb  # Interactive Colab notebook for cloud GPU training
│
├── docs/                             # Documentation & Training History
│   └── training-logs/
│       ├── colab1.md                 # Initial training run logs (pre-fix)
│       ├── colab2.md                 # Second training run logs (debugging)
│       └── colab3.md                 # Final training run logs (50 epochs Fusion + 25 epochs U-Net)
│
├── train.py                          # Training loop: AMP, OneCycleLR, global confusion matrix mIoU
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # CUDA-enabled container definition
├── docker-compose.yml                # Multi-container service orchestrator
├── .env                              # Environment variables
├── .gitignore                        # Git exclusions (datasets, weights, node_modules)
└── README.md                         # Project documentation (this file)
```

---

### 0. Model Checkpoints & Pre-trained Weights

Pre-trained model weights for both the Siamese Fusion Network and Baseline U-Net can be downloaded directly from Google Drive:

[![Google Drive](https://img.shields.io/badge/Google%20Drive-Download%20Weights-4285F4.svg?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1FYg7WhwLALRmfS8VdveY_0WzItsQ2L22?usp=sharing)

🔗 **Direct Download Link:** [Google Drive Weights Folder](https://drive.google.com/drive/folders/1FYg7WhwLALRmfS8VdveY_0WzItsQ2L22?usp=sharing)

#### Included Checkpoints:
- `fusion_net_best.pth` (~584 MB) — Novel Siamese Cross-Attention Network (O3)
- `unet_net_best.pth` (~572 MB) — Baseline ResNet34 U-Net (O2)

> **Setup:** Download the `.pth` files and place them into the `weights/` directory:
> ```
> CAPSTONE-PROJECT/
> └── weights/
>     ├── fusion_net_best.pth
>     └── unet_net_best.pth
> ```

---

### 1. Training on Google Colab (T4 GPU)

You can also train or fine-tune both models in Google Colab with a free **NVIDIA Tesla T4 GPU**:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isuryaprakashh/Capstone-Flood-Mapping/blob/main/notebooks/SpaceNet8_Colab_Training.ipynb)

Open [`notebooks/SpaceNet8_Colab_Training.ipynb`](notebooks/SpaceNet8_Colab_Training.ipynb) for step-by-step interactive training, automated Google Drive checkpoint backup, evaluation benchmarks, and visual qualitative inspections.

---

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/isuryaprakashh/Capstone-Flood-Mapping.git
cd Capstone-Flood-Mapping

# Install Python dependencies
pip install -r requirements.txt

# Install Frontend dependencies
cd frontend
npm install
cd ..
```

---

### 3. Running the Web Platform & API

#### Option A: Running with Trained Model Weights (Real Inference)

1. **Start the FastAPI Backend:**
   - **PowerShell (Windows):**
     ```powershell
     $env:USE_STUB="false"
     $env:MODEL_TYPE="fusion"
     $env:O3_FUSION_WEIGHTS="./weights/fusion_net_best.pth"
     python backend/main.py
     ```
   - **Linux / macOS (Bash):**
     ```bash
     USE_STUB=false MODEL_TYPE=fusion O3_FUSION_WEIGHTS=./weights/fusion_net_best.pth python backend/main.py
     ```
   - **Command Prompt (CMD):**
     ```cmd
     set USE_STUB=false
     set MODEL_TYPE=fusion
     set O3_FUSION_WEIGHTS=./weights/fusion_net_best.pth
     python backend/main.py
     ```

2. **Start the React Frontend Dashboard (in a second terminal):**
   ```bash
   cd frontend
   npm run dev
   ```
   - Frontend UI: `http://localhost:3000`
   - Backend API Docs: `http://localhost:8000/api/docs`

#### Option B: Running in Stub / Development Mode (Without GPU / Weights)
```bash
# Terminal 1: Backend
python backend/main.py

# Terminal 2: Frontend
cd frontend
npm run dev
```

---

### 4. Data Preprocessing & Mask Generation

```bash
# Generate 5-channel segmentation masks from GeoJSON & reference CSVs
python scripts/tile_images.py --data-root ./CP1-DATASET --output-dir ./processed
```

---

### 5. Local / Cloud Model Training

```bash
# Train the Novel Siamese Cross-Attention Fusion Network (O3)
python train.py --model fusion --data-root ./CP1-DATASET --epochs 50 --batch-size 8 --amp

# Train the Baseline U-Net (O2)
python train.py --model unet --data-root ./CP1-DATASET --epochs 35 --batch-size 8 --amp
```

---

### 6. Model Evaluation & Benchmark Comparison

```bash
python scripts/eval.py \
  --data-root ./CP1-DATASET \
  --unet-ckpt weights/unet_net_best.pth \
  --fusion-ckpt weights/fusion_net_best.pth
```

---

## 🧪 Testing Suite

Run the full automated test suite:

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
