#!/bin/bash
# ==============================================================================
# SpaceNet 8 Flood Mapping — Direct HPC Shell Execution Script
# Use this when running directly inside an interactive GPU node / tmux session
# ==============================================================================

set -e

echo "=========================================================="
echo "🚀 SpaceNet 8 Complete Training Pipeline"
echo "=========================================================="

mkdir -p logs checkpoints processed/masks

# Step 1: Preprocess Masks
echo ""
echo "📦 [1/3] Generating multi-channel binary segmentation masks..."
python3 scripts/tile_images.py --data-root ./CP1-DATASET --output-dir ./processed

# Step 2: Train Siamese Fusion Network (O3)
echo ""
echo "🧠 [2/3] Training Siamese Cross-Attention Network (O3 Novel Model)..."
python3 train.py \
    --model fusion \
    --data-root ./CP1-DATASET \
    --mask-dir ./processed/masks \
    --epochs 100 \
    --batch-size 8 \
    --lr 0.001 \
    --amp \
    --checkpoint-dir ./checkpoints \
    --log-dir ./logs

# Step 3: Train Baseline U-Net (O2)
echo ""
echo "🔬 [3/3] Training Baseline U-Net Model (O2 Baseline)..."
python3 train.py \
    --model unet \
    --data-root ./CP1-DATASET \
    --mask-dir ./processed/masks \
    --epochs 50 \
    --batch-size 8 \
    --lr 0.001 \
    --amp \
    --checkpoint-dir ./checkpoints \
    --log-dir ./logs

# Step 4: Run Evaluation & Generate Benchmark Table
echo ""
echo "📊 Evaluating models and generating comparison report..."
python3 eval.py \
    --data-root ./CP1-DATASET \
    --mask-dir ./processed/masks \
    --unet-ckpt ./checkpoints/unet_net_best.pth \
    --fusion-ckpt ./checkpoints/fusion_net_best.pth \
    --output ./logs/final_eval_benchmark.json

echo ""
echo "=========================================================="
echo "✅ All Training & Evaluations Complete!"
echo "Checkpoints saved in ./checkpoints/"
echo "=========================================================="
