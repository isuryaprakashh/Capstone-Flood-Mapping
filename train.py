#!/usr/bin/env python3
"""
Training Script for SpaceNet 8 Flood Mapping Models.

Supports:
  - U-Net Baseline and Siamese Cross-Attention Fusion models
  - Mixed-precision training (AMP) with gradient scaling
  - OneCycleLR learning rate scheduling
  - W&B / CSV metric logging
  - Best-IoU checkpoint saving
  - Multi-GPU DataParallel
  - Early stopping

Usage:
    # Train baseline U-Net
    python train.py --model unet --data-root ./CP1-DATASET --epochs 50

    # Train Siamese Fusion model
    python train.py --model fusion --data-root ./CP1-DATASET --epochs 100

    # Resume training from checkpoint
    python train.py --model fusion --resume checkpoints/fusion_net_best.pth
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from data.dataset import create_dataloaders, NUM_CLASSES
from losses.combined_loss import CombinedLoss
from models import build_model, count_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Metrics ────────────────────────────────────────────────────────────────

CLASS_NAMES = ["building", "road", "flood", "flooded_building", "flooded_road"]


def compute_metrics(
    preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> dict:
    """
    Compute per-class IoU, F1, Precision, Recall.

    Args:
        preds: Predicted logits (B, C, H, W).
        targets: Ground truth masks (B, C, H, W).
        threshold: Binarization threshold.

    Returns:
        Dictionary of metric name → value.
    """
    with torch.no_grad():
        probs = torch.sigmoid(preds)
        binary = (probs > threshold).float()

        metrics = {}
        ious = []

        for c in range(min(preds.shape[1], len(CLASS_NAMES))):
            pred_c = binary[:, c].flatten()
            targ_c = targets[:, c].flatten()

            tp = (pred_c * targ_c).sum().item()
            fp = (pred_c * (1 - targ_c)).sum().item()
            fn = ((1 - pred_c) * targ_c).sum().item()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            iou = tp / (tp + fp + fn + 1e-8)

            name = CLASS_NAMES[c]
            metrics[f"iou_{name}"] = iou
            metrics[f"f1_{name}"] = f1
            metrics[f"precision_{name}"] = precision
            metrics[f"recall_{name}"] = recall
            ious.append(iou)

        metrics["mean_iou"] = np.mean(ious)
        metrics["mean_f1"] = np.mean(
            [metrics[f"f1_{n}"] for n in CLASS_NAMES[: preds.shape[1]]]
        )

    return metrics


# ─── Training Loop ──────────────────────────────────────────────────────────


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: CombinedLoss,
    optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
) -> dict:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_focal = 0.0
    running_bce = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        pre_img = batch["pre_image"].to(device, non_blocking=True)
        post_img = batch["post_image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=scaler.is_enabled()):
            logits = model(pre_img, post_img)
            loss_dict = criterion(logits, mask)
            loss = loss_dict["total"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        running_dice += loss_dict["dice"].item()
        running_focal += loss_dict["focal"].item()
        running_bce += loss_dict["bce"].item()
        num_batches += 1

        if (batch_idx + 1) % 20 == 0:
            avg = running_loss / num_batches
            logger.info(
                f"  Epoch {epoch} [{batch_idx + 1}/{len(loader)}] "
                f"Loss: {avg:.4f}"
            )

    n = max(num_batches, 1)
    return {
        "train_loss": running_loss / n,
        "train_dice_loss": running_dice / n,
        "train_focal_loss": running_focal / n,
        "train_bce_loss": running_bce / n,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: CombinedLoss,
    device: torch.device,
) -> dict:
    """Run validation."""
    model.eval()
    running_loss = 0.0
    all_metrics = []
    num_batches = 0

    for batch in loader:
        pre_img = batch["pre_image"].to(device, non_blocking=True)
        post_img = batch["post_image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        logits = model(pre_img, post_img)
        loss_dict = criterion(logits, mask)

        running_loss += loss_dict["total"].item()
        all_metrics.append(compute_metrics(logits, mask))
        num_batches += 1

    n = max(num_batches, 1)
    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    avg_metrics["val_loss"] = running_loss / n
    return avg_metrics


# ─── Logging ────────────────────────────────────────────────────────────────


class CSVLogger:
    """Simple CSV metric logger."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None
        self.writer = None

    def log(self, metrics: dict):
        if self.file is None:
            self.file = open(self.filepath, "w", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=list(metrics.keys()))
            self.writer.writeheader()

        self.writer.writerow(
            {k: f"{v:.6f}" if isinstance(v, float) else v for k, v in metrics.items()}
        )
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


def try_wandb_log(metrics: dict, step: int):
    """Log to W&B if available."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass


# ─── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train SpaceNet 8 Flood Mapping Model")

    # Model
    parser.add_argument("--model", type=str, default="fusion", choices=["unet", "fusion"])
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")

    # Data
    parser.add_argument("--data-root", type=str, default="./CP1-DATASET")
    parser.add_argument("--mask-dir", type=str, default=None)
    parser.add_argument("--val-region", type=str, default="Germany_Training_Public")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")

    # Loss
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--focal-weight", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=0.5)

    # Output
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--wandb-project", type=str, default="capstone-flood-mapping")
    parser.add_argument("--no-wandb", action="store_true", default=False)

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # W&B init
    if not args.no_wandb:
        try:
            import wandb

            wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"{args.model}_{time.strftime('%Y%m%d_%H%M%S')}",
            )
        except (ImportError, Exception) as e:
            logger.warning(f"W&B not available: {e}")

    # Data
    logger.info("Creating DataLoaders...")
    loaders = create_dataloaders(
        data_root=args.data_root,
        mask_dir=args.mask_dir,
        batch_size=args.batch_size,
        tile_size=args.tile_size,
        num_workers=args.num_workers,
        val_region=args.val_region,
    )

    # Model
    logger.info(f"Building model: {args.model}")
    if args.model == "unet":
        model = build_model(
            "unet",
            in_channels=6,
            num_classes=NUM_CLASSES,
            pretrained=args.pretrained,
        )
    else:
        model = build_model(
            "fusion",
            in_channels=3,
            num_classes=NUM_CLASSES,
            pretrained=args.pretrained,
        )

    params = count_parameters(model)
    logger.info(f"Parameters: {params['total_millions']}M ({params['trainable']} trainable)")

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
        model = nn.DataParallel(model)

    model = model.to(device)

    # Loss, Optimizer, Scheduler
    criterion = CombinedLoss(
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight,
        bce_weight=args.bce_weight,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(loaders["train"]),
        pct_start=0.1,
        anneal_strategy="cos",
    )

    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    # Resume checkpoint
    start_epoch = 0
    best_iou = 0.0

    if args.resume and Path(args.resume).exists():
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model_state = ckpt.get("model_state_dict", ckpt)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_iou = ckpt.get("best_iou", 0.0)
        logger.info(f"Resumed at epoch {start_epoch}, best IoU: {best_iou:.4f}")

    # CSV Logger
    csv_logger = CSVLogger(
        os.path.join(args.log_dir, f"{args.model}_metrics.csv")
    )

    # Training
    patience_counter = 0

    logger.info("=" * 60)
    logger.info(f"Starting training: {args.model} for {args.epochs} epochs")
    logger.info("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device, epoch
        )

        # Step scheduler
        scheduler.step()

        # Validate
        val_metrics = validate(model, loaders["val"], criterion, device)

        epoch_time = time.time() - epoch_start
        current_iou = val_metrics.get("mean_iou", 0.0)

        # Log
        log_row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **train_metrics,
            **val_metrics,
            "epoch_time_s": round(epoch_time, 1),
        }

        csv_logger.log(log_row)
        try_wandb_log(log_row, step=epoch)

        logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"mIoU: {current_iou:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best checkpoint
        if current_iou > best_iou:
            best_iou = current_iou
            patience_counter = 0

            model_state = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )

            ckpt_path = os.path.join(
                args.checkpoint_dir, f"{args.model}_net_best.pth"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_iou": best_iou,
                    "args": vars(args),
                    "metrics": val_metrics,
                },
                ckpt_path,
            )
            logger.info(f"  ★ New best model saved: mIoU={best_iou:.4f} → {ckpt_path}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    csv_logger.close()
    logger.info("=" * 60)
    logger.info(f"Training complete. Best mIoU: {best_iou:.4f}")
    logger.info(f"Best checkpoint: {args.checkpoint_dir}/{args.model}_net_best.pth")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
