#!/usr/bin/env python3
"""
Measure residual pre/post misalignment across the SpaceNet 8 training AOIs.

For every labelled tile pair, the post-event image is resampled onto the
pre-event grid using the GeoTIFF georeferencing, then the remaining shift is
estimated with block-wise phase correlation. Results are written to a CSV
and summarised per AOI (used for the dataset analysis in the paper).

Usage:
    python scripts/analyze_alignment.py --data-root ./CP1-DATASET \
        --output docs/analysis/alignment_stats.csv
"""

import argparse
import csv
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataloader.align import estimate_shift_blocks, load_aligned_pair

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

AOIS = ["Germany_Training_Public", "Louisiana-East_Training_Public"]


def list_pairs(data_root: Path) -> list[dict]:
    """List every (pre, post) pair, including second post images."""
    pairs = []
    for aoi in AOIS:
        aoi_dir = data_root / aoi
        mapping = aoi_dir / f"{aoi}_label_image_mapping.csv"
        if not mapping.exists():
            logger.warning(f"Missing {mapping}")
            continue
        with open(mapping) as f:
            for row in csv.DictReader(f):
                for idx, key in enumerate(["post-event image 1", "post-event image 2"], start=1):
                    post = (row.get(key) or "").strip()
                    if not post:
                        continue
                    pairs.append({
                        "aoi": aoi,
                        "tile": row["label"].replace(".geojson", ""),
                        "post_index": idx,
                        "pre_path": str(aoi_dir / "PRE-event" / row["pre-event image"].strip()),
                        "post_path": str(aoi_dir / "POST-event" / post),
                    })
    return pairs


def analyze_pair(pair: dict) -> dict:
    """Resample one pair and estimate its residual shift."""
    try:
        p = load_aligned_pair(pair["pre_path"], pair["post_path"])
        r = estimate_shift_blocks(p["pre"], p["post"], p["valid"])
        return {
            **{k: pair[k] for k in ("aoi", "tile", "post_index")},
            "pre_gsd_m": round(p["pre_gsd_m"], 3),
            "post_gsd_m": round(p["post_gsd_m"], 3),
            "valid_fraction": round(float(p["valid"].mean()), 4),
            "dy_px": round(r["dy"], 2),
            "dx_px": round(r["dx"], 2),
            "shift_px": round(r["magnitude"], 2),
            "shift_m": round(r["magnitude"] * p["pre_gsd_m"], 2),
            "spread_px": round(r["spread"], 2),
            "confident_blocks": r["n_confident"],
            "blocks": r["n_blocks"],
        }
    except Exception as e:
        logger.error(f"{pair['tile']}: {e}")
        return None


def summarize(rows: list[dict]):
    """Print per-AOI misalignment statistics."""
    for aoi in AOIS:
        sub = [r for r in rows if r["aoi"] == aoi]
        est = [r for r in sub if not np.isnan(r["shift_px"])]
        if not sub:
            continue
        shift = np.array([r["shift_px"] for r in est])
        meters = np.array([r["shift_m"] for r in est])
        spread = np.array([r["spread_px"] for r in est])
        logger.info(
            f"{aoi}: {len(sub)} pairs, {len(est)} with a confident estimate | "
            f"shift px median {np.median(shift):.1f}, p90 {np.percentile(shift, 90):.1f}, "
            f"max {shift.max():.1f} | meters median {np.median(meters):.2f}, "
            f"p90 {np.percentile(meters, 90):.2f} | >4px: {100 * np.mean(shift > 4):.0f}%, "
            f">8px: {100 * np.mean(shift > 8):.0f}% | non-rigid spread median {np.median(spread):.1f}px"
        )


def main():
    parser = argparse.ArgumentParser(description="Measure SpaceNet 8 pre/post misalignment")
    parser.add_argument("--data-root", type=str, default="./CP1-DATASET")
    parser.add_argument("--output", type=str, default="docs/analysis/alignment_stats.csv")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    pairs = list_pairs(Path(args.data_root))
    logger.info(f"Analyzing {len(pairs)} pre/post pairs with {args.workers} workers")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(analyze_pair, pairs, chunksize=4), start=1):
            if row is not None:
                rows.append(row)
            if i % 100 == 0:
                logger.info(f"  {i}/{len(pairs)} pairs")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} rows to {out}")

    summarize(rows)


if __name__ == "__main__":
    main()
