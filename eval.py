#!/usr/bin/env python3
"""
Evaluation Script — Compare Baseline U-Net vs Siamese Fusion Model.

Loads trained checkpoints, runs inference on the validation/test set,
and generates a comparison table with per-class metrics.

Usage:
    python eval.py --data-root ./CP1-DATASET \
                   --unet-ckpt checkpoints/unet_net_best.pth \
                   --fusion-ckpt checkpoints/fusion_net_best.pth
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data.dataset import create_dataloaders, NUM_CLASSES
from models import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CLASS_NAMES = ["building", "road", "flood", "flooded_building", "flooded_road"]


def compute_detailed_metrics(
    preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> dict:
    """Compute per-class and aggregate metrics."""
    with torch.no_grad():
        probs = torch.sigmoid(preds)
        binary = (probs > threshold).float()

        metrics = {}

        for c in range(min(preds.shape[1], len(CLASS_NAMES))):
            pred_c = binary[:, c].flatten()
            targ_c = targets[:, c].flatten()

            tp = (pred_c * targ_c).sum().item()
            fp = (pred_c * (1 - targ_c)).sum().item()
            fn = ((1 - pred_c) * targ_c).sum().item()
            tn = ((1 - pred_c) * (1 - targ_c)).sum().item()

            name = CLASS_NAMES[c]
            metrics[f"iou_{name}"] = tp / (tp + fp + fn + 1e-8)
            metrics[f"f1_{name}"] = 2 * tp / (2 * tp + fp + fn + 1e-8)
            metrics[f"precision_{name}"] = tp / (tp + fp + 1e-8)
            metrics[f"recall_{name}"] = tp / (tp + fn + 1e-8)
            metrics[f"accuracy_{name}"] = (tp + tn) / (tp + tn + fp + fn + 1e-8)

        ious = [metrics[f"iou_{n}"] for n in CLASS_NAMES[: preds.shape[1]]]
        f1s = [metrics[f"f1_{n}"] for n in CLASS_NAMES[: preds.shape[1]]]
        metrics["mean_iou"] = np.mean(ious)
        metrics["mean_f1"] = np.mean(f1s)

    return metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader,
    device: torch.device,
    model_name: str = "model",
) -> dict:
    """
    Run full evaluation on a dataset.

    Returns:
        Aggregated metrics dictionary.
    """
    model.eval()
    all_metrics = []

    for batch_idx, batch in enumerate(loader):
        pre_img = batch["pre_image"].to(device, non_blocking=True)
        post_img = batch["post_image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        logits = model(pre_img, post_img)
        metrics = compute_detailed_metrics(logits, mask)
        all_metrics.append(metrics)

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  [{model_name}] Evaluated {batch_idx + 1}/{len(loader)} batches")

    # Average all metrics
    avg = {}
    if all_metrics:
        for key in all_metrics[0]:
            avg[key] = float(np.mean([m[key] for m in all_metrics]))

    return avg


def load_model_from_checkpoint(
    model_type: str,
    checkpoint_path: str,
    device: torch.device,
) -> nn.Module:
    """Load a model from a checkpoint file."""
    if model_type == "unet":
        model = build_model("unet", in_channels=6, num_classes=NUM_CLASSES, pretrained=False)
    else:
        model = build_model("fusion", in_channels=3, num_classes=NUM_CLASSES, pretrained=False)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Handle DataParallel state dict
    new_state = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state[new_key] = v

    model.load_state_dict(new_state, strict=False)
    model = model.to(device)
    model.eval()

    logger.info(
        f"Loaded {model_type} from {checkpoint_path} "
        f"(epoch {ckpt.get('epoch', '?')}, best_iou={ckpt.get('best_iou', '?')})"
    )

    return model


def print_comparison_table(results: dict[str, dict]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("MODEL COMPARISON — SpaceNet 8 Flood Mapping")
    print("=" * 90)

    # Header
    models = list(results.keys())
    header = f"{'Metric':<25}"
    for m in models:
        header += f"| {m:>20} "
    header += "| {'Δ (Fusion - UNet)':>20}" if len(models) >= 2 else ""
    print(header)
    print("-" * 90)

    # Per-class metrics
    for cls in CLASS_NAMES:
        for metric_type in ["iou", "f1"]:
            key = f"{metric_type}_{cls}"
            row = f"{key:<25}"
            values = []
            for m in models:
                val = results[m].get(key, 0.0)
                values.append(val)
                row += f"| {val:>19.4f} "

            if len(values) >= 2:
                delta = values[-1] - values[0]
                sign = "+" if delta >= 0 else ""
                row += f"| {sign}{delta:>18.4f} "

            print(row)
        print("-" * 90)

    # Aggregate metrics
    for key in ["mean_iou", "mean_f1"]:
        row = f"{'★ ' + key:<25}"
        values = []
        for m in models:
            val = results[m].get(key, 0.0)
            values.append(val)
            row += f"| {val:>19.4f} "

        if len(values) >= 2:
            delta = values[-1] - values[0]
            sign = "+" if delta >= 0 else ""
            row += f"| {sign}{delta:>18.4f} "

        print(row)

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Flood Mapping Models")

    parser.add_argument("--data-root", type=str, default="./CP1-DATASET")
    parser.add_argument("--mask-dir", type=str, default=None)
    parser.add_argument("--unet-ckpt", type=str, default="checkpoints/unet_net_best.pth")
    parser.add_argument("--fusion-ckpt", type=str, default="checkpoints/fusion_net_best.pth")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--output", type=str, default="eval_results.json")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Create DataLoader
    loaders = create_dataloaders(
        data_root=args.data_root,
        mask_dir=args.mask_dir,
        batch_size=args.batch_size,
        tile_size=args.tile_size,
        num_workers=2,
    )
    eval_loader = loaders[args.split]
    logger.info(f"Evaluating on {args.split} set: {len(eval_loader.dataset)} samples")

    results = {}

    # Evaluate U-Net baseline
    if Path(args.unet_ckpt).exists():
        logger.info("\n--- Evaluating U-Net Baseline ---")
        unet = load_model_from_checkpoint("unet", args.unet_ckpt, device)
        results["UNet Baseline"] = evaluate_model(unet, eval_loader, device, "U-Net")
        del unet
        torch.cuda.empty_cache() if device.type == "cuda" else None
    else:
        logger.warning(f"U-Net checkpoint not found: {args.unet_ckpt}")

    # Evaluate Siamese Fusion
    if Path(args.fusion_ckpt).exists():
        logger.info("\n--- Evaluating Siamese Fusion ---")
        fusion = load_model_from_checkpoint("fusion", args.fusion_ckpt, device)
        results["Siamese Fusion"] = evaluate_model(fusion, eval_loader, device, "Fusion")
        del fusion
        torch.cuda.empty_cache() if device.type == "cuda" else None
    else:
        logger.warning(f"Fusion checkpoint not found: {args.fusion_ckpt}")

    if not results:
        logger.error("No model checkpoints found. Train models first.")
        sys.exit(1)

    # Print comparison table
    print_comparison_table(results)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
