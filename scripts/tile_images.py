#!/usr/bin/env python3
"""
SpaceNet 8 Preprocessing Script — Tile Images & Generate Masks.

Runs the full data preprocessing pipeline:
  1. Parse dataset directories
  2. (Optional) Reproject images to unified CRS
  3. Generate multi-channel segmentation masks from annotations
  4. Create train/val/test split manifest

Usage:
    python scripts/tile_images.py --data-root ./CP1-DATASET --output-dir ./processed
    python scripts/tile_images.py --data-root ./CP1-DATASET --masks-only
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.mask_generator import batch_generate_masks
from data.dataset import build_sample_manifest, geographic_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="SpaceNet 8 Data Preprocessing Pipeline"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./CP1-DATASET",
        help="Root directory containing SpaceNet 8 dataset splits",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--masks-only",
        action="store_true",
        help="Only generate masks, skip tiling",
    )
    parser.add_argument(
        "--road-buffer",
        type=float,
        default=4.0,
        help="Road buffer width in pixels for rasterization",
    )
    parser.add_argument(
        "--val-region",
        type=str,
        default="Germany_Training_Public",
        help="Region to use as validation holdout",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Fraction of training data for test set",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    if not data_root.exists():
        logger.error(f"Data root does not exist: {data_root}")
        sys.exit(1)

    # Discover dataset directories
    dataset_dirs = []
    for child in sorted(data_root.iterdir()):
        if child.is_dir() and (child / "PRE-event").exists():
            dataset_dirs.append(child)
            logger.info(f"Found dataset: {child.name}")

    if not dataset_dirs:
        logger.error(f"No SpaceNet 8 dataset directories found in {data_root}")
        sys.exit(1)

    # Step 1: Generate masks
    mask_dir = output_dir / "masks"
    logger.info("=" * 60)
    logger.info("STEP 1: Generating segmentation masks")
    logger.info("=" * 60)

    total_masks = 0
    for dataset_dir in dataset_dirs:
        if not (dataset_dir / "annotations").exists():
            logger.info(f"  Skipping {dataset_dir.name} (no annotations — test set)")
            continue

        region_mask_dir = mask_dir / dataset_dir.name
        logger.info(f"  Processing {dataset_dir.name}...")

        generated = batch_generate_masks(
            dataset_dir=dataset_dir,
            output_dir=region_mask_dir,
            road_buffer_pixels=args.road_buffer,
        )
        total_masks += len(generated)
        logger.info(f"  → Generated {len(generated)} masks")

    logger.info(f"Total masks generated: {total_masks}")

    # Step 2: Build manifest and create splits
    logger.info("=" * 60)
    logger.info("STEP 2: Building data manifest & splits")
    logger.info("=" * 60)

    # Only include directories with annotations for training
    train_dirs = [
        d for d in dataset_dirs if (d / "annotations").exists()
    ]

    samples = build_sample_manifest(train_dirs, mask_dir=str(mask_dir))
    logger.info(f"Total samples in manifest: {len(samples)}")

    train_samples, val_samples, test_samples = geographic_split(
        samples,
        val_region=args.val_region,
        test_fraction=args.test_fraction,
    )

    # Save manifest files
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }

    for split_name, split_samples in splits.items():
        manifest_path = manifest_dir / f"{split_name}_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(split_samples, f, indent=2)
        logger.info(f"  Saved {split_name} manifest: {len(split_samples)} samples → {manifest_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Masks directory:    {mask_dir}")
    logger.info(f"  Manifests directory: {manifest_dir}")
    logger.info(f"  Train samples: {len(train_samples)}")
    logger.info(f"  Val samples:   {len(val_samples)}")
    logger.info(f"  Test samples:  {len(test_samples)}")
    logger.info("")
    logger.info("Next step: Run training with:")
    logger.info(f"  python train.py --data-root {data_root} --mask-dir {mask_dir}")


if __name__ == "__main__":
    main()
