#!/bin/bash
# ==============================================================================
# SpaceNet 8 Flood Mapping — Quick Google Colab Setup & Run Script (T4 GPU)
# ==============================================================================

set -e

echo "=========================================================="
echo "🚀 SpaceNet 8 Training Pipeline for Google Colab (T4 GPU)"
echo "=========================================================="

# 1. Install dependencies
echo "📦 Installing required packages..."
pip install -q rasterio geopandas shapely albumentations timm wandb networkx scikit-image pytest

# 2. Create directories
mkdir -p logs checkpoints processed/masks

# 3. Check GPU
python3 -c "import torch; print(f'GPU Active: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None (CPU)\"}')"

# 4. Download SpaceNet 8 Datasets from AWS S3 (if not present)
mkdir -p CP1-DATASET
if [ ! -d "CP1-DATASET/Germany_Training_Public" ]; then
    echo "⬇️ Downloading SpaceNet 8 Datasets from AWS S3..."
    wget -c -q --show-progress https://spacenet-dataset.s3.amazonaws.com/spacenet/SN8_floods/tarballs/Germany_Training_Public.tar.gz -O CP1-DATASET/Germany_Training_Public.tar.gz
    wget -c -q --show-progress https://spacenet-dataset.s3.amazonaws.com/spacenet/SN8_floods/tarballs/Louisiana-East_Training_Public.tar.gz -O CP1-DATASET/Louisiana-East_Training_Public.tar.gz
    wget -c -q --show-progress https://spacenet-dataset.s3.amazonaws.com/spacenet/SN8_floods/tarballs/Louisiana-West_Test_Public.tar.gz -O CP1-DATASET/Louisiana-West_Test_Public.tar.gz

    echo "📦 Extracting datasets..."
    tar -xzf CP1-DATASET/Germany_Training_Public.tar.gz -C CP1-DATASET/
    tar -xzf CP1-DATASET/Louisiana-East_Training_Public.tar.gz -C CP1-DATASET/
    tar -xzf CP1-DATASET/Louisiana-West_Test_Public.tar.gz -C CP1-DATASET/
    rm -f CP1-DATASET/*.tar.gz
    echo "✅ Datasets downloaded and extracted!"
else
    echo "✅ Dataset already found in CP1-DATASET."
fi

# 5. Generate Masks if not already generated
if [ ! -d "processed/masks/Germany_Training_Public" ]; then
    echo "🗺️ Generating 5-channel segmentation masks..."
    python3 scripts/tile_images.py --data-root ./CP1-DATASET --output-dir ./processed --val-region Germany_Training_Public --road-buffer 4.0
else
    echo "✅ Masks already present in ./processed/masks. Skipping generation."
fi

# 5. Train Siamese Cross-Attention Network (O3)
echo "🧠 Training Siamese Cross-Attention Network (40 epochs with OneCycleLR & AMP)..."
python3 train.py \
    --model fusion \
    --data-root ./CP1-DATASET \
    --mask-dir ./processed/masks \
    --epochs 40 \
    --batch-size 8 \
    --num-workers 2 \
    --lr 0.001 \
    --amp \
    --patience 12 \
    --checkpoint-dir ./checkpoints \
    --log-dir ./logs \
    --no-wandb

# 6. Train Baseline U-Net (O2)
echo "🔬 Training Baseline U-Net (25 epochs)..."
python3 train.py \
    --model unet \
    --data-root ./CP1-DATASET \
    --mask-dir ./processed/masks \
    --epochs 25 \
    --batch-size 8 \
    --num-workers 2 \
    --lr 0.001 \
    --amp \
    --patience 8 \
    --checkpoint-dir ./checkpoints \
    --log-dir ./logs \
    --no-wandb

# 7. Evaluate with Test-Time Augmentation (TTA) & Overlap Attribution
echo "📊 Evaluating models with TTA & SpaceNet 8 Overlap Heuristics..."
python3 eval.py \
    --data-root ./CP1-DATASET \
    --mask-dir ./processed/masks \
    --unet-ckpt ./checkpoints/unet_net_best.pth \
    --fusion-ckpt ./checkpoints/fusion_net_best.pth \
    --tta \
    --heuristics \
    --output ./logs/final_eval_benchmark.json

echo "=========================================================="
echo "🎉 Colab Training & Benchmark Complete!"
echo "Checkpoints: ./checkpoints/"
echo "Logs:        ./logs/"
echo "=========================================================="
