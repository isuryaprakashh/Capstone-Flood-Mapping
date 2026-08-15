#!/bin/bash
# ==============================================================================
# SpaceNet 8 Flood Mapping — HPC SLURM Batch Job Script
# ==============================================================================
#SBATCH --job-name=flood_mapping_training
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1               # Request 1 GPU (or change to gpu:2/4)
#SBATCH --mem=32G                  # Memory requirement
#SBATCH --time=24:00:00            # Max runtime (hh:mm:ss)
#SBATCH --partition=gpu            # Adjust to your cluster's GPU partition name

echo "=================================================================="
echo "Job ID:            $SLURM_JOB_ID"
echo "Running on node:   $(hostname)"
echo "Starting at:       $(date)"
echo "CUDA Devices:      $CUDA_VISIBLE_DEVICES"
echo "=================================================================="

# 1. Environment Activation (Uncomment and adjust for your HPC module/conda setup)
# module load cuda/12.1
# module load python/3.10
# source ~/miniconda3/bin/activate flood_env

# 2. Make directories
mkdir -p logs checkpoints processed/masks

# 3. Step 1: Preprocess Masks (if not already done)
if [ ! -d "processed/masks/Germany_Training_Public" ]; then
    echo ">>> Step 1: Generating segmentation masks from SpaceNet 8 annotations..."
    python3 scripts/tile_images.py --data-root ./CP1-DATASET --output-dir ./processed
else
    echo ">>> Step 1: Masks already generated. Skipping."
fi

# 4. Step 2: Train Novel Siamese Cross-Attention Fusion Network (O3)
echo ">>> Step 2: Training Siamese Cross-Attention Network (100 epochs)..."
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

# 5. Step 3: Train Baseline U-Net (O2) for comparative benchmark
echo ">>> Step 3: Training Baseline U-Net Model (50 epochs)..."
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

# 6. Step 4: Run Comparative Evaluation
echo ">>> Step 4: Evaluating both models & generating benchmark table..."
python3 eval.py \
    --data-root ./CP1-DATASET \
    --mask-dir ./processed/masks \
    --unet-ckpt ./checkpoints/unet_net_best.pth \
    --fusion-ckpt ./checkpoints/fusion_net_best.pth \
    --output ./logs/final_eval_benchmark.json

echo "=================================================================="
echo "Completed at: $(date)"
echo "Best models saved in ./checkpoints/"
echo "=================================================================="
