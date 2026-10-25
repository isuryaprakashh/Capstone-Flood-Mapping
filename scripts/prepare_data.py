#!/usr/bin/env python3
"""
Prepare SpaceNet 8 for training: co-registered image pairs, targets,
ground-truth vectors and spatially blocked protocol splits.

Usage:
    python scripts/prepare_data.py --data-root ./CP1-DATASET --out-dir ./processed
    python scripts/prepare_data.py --data-root ./CP1-DATASET --out-dir ./processed_smoke --limit 12
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataloader.prepare import prepare_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="Prepare SpaceNet 8 for training")
    parser.add_argument("--data-root", type=str, default="./CP1-DATASET")
    parser.add_argument("--out-dir", type=str, default="./processed")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--include-unlabelled", action="store_true",
                        help="Also prepare Louisiana-West (images only, no labels)")
    parser.add_argument("--limit", type=int, default=None, help="Max tiles per AOI (smoke tests)")
    parser.add_argument("--seed", type=int, default=0, help="Fold assignment seed")
    args = parser.parse_args()

    manifest = prepare_dataset(
        args.data_root, args.out_dir,
        workers=args.workers,
        include_unlabelled=args.include_unlabelled,
        limit=args.limit,
        seed=args.seed,
    )
    if manifest["n_failed"]:
        logging.error(f"{manifest['n_failed']} tiles failed — see errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
