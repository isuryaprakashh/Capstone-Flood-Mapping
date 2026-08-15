"""
Image Tiling Utilities for SpaceNet 8 GeoTIFF Imagery.

Tiles large satellite images into smaller 512×512 patches suitable for
model training, with configurable overlap and edge padding.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import rasterio
from rasterio.windows import Window

logger = logging.getLogger(__name__)

# Default tile configuration
DEFAULT_TILE_SIZE = 512
DEFAULT_OVERLAP = 64


def compute_tile_windows(
    img_width: int,
    img_height: int,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Window]:
    """
    Compute a grid of rasterio Windows covering the entire image.

    Args:
        img_width: Image width in pixels.
        img_height: Image height in pixels.
        tile_size: Size of each square tile.
        overlap: Overlap between adjacent tiles in pixels.

    Returns:
        List of rasterio Window objects.
    """
    stride = tile_size - overlap
    windows = []

    for y in range(0, img_height, stride):
        for x in range(0, img_width, stride):
            # Clamp window to image bounds
            win_width = min(tile_size, img_width - x)
            win_height = min(tile_size, img_height - y)
            windows.append(Window(x, y, win_width, win_height))

    return windows


def tile_geotiff(
    src_path: Union[str, Path],
    dst_dir: Union[str, Path],
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    min_data_fraction: float = 0.1,
    pad_value: float = 0.0,
) -> list[Path]:
    """
    Tile a GeoTIFF image into smaller patches.

    Args:
        src_path: Path to the source GeoTIFF.
        dst_dir: Output directory for tile files.
        tile_size: Size of each square tile.
        overlap: Pixel overlap between adjacent tiles.
        min_data_fraction: Minimum fraction of non-zero pixels to keep a tile.
        pad_value: Value to use for padding edge tiles.

    Returns:
        List of paths to generated tile files.
    """
    src_path = Path(src_path)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []

    with rasterio.open(src_path) as src:
        windows = compute_tile_windows(
            src.width, src.height, tile_size, overlap
        )

        for idx, window in enumerate(windows):
            # Read the window data
            data = src.read(window=window)  # (C, H, W)

            # Pad if the tile is smaller than tile_size
            _, h, w = data.shape
            if h < tile_size or w < tile_size:
                padded = np.full(
                    (data.shape[0], tile_size, tile_size),
                    pad_value,
                    dtype=data.dtype,
                )
                padded[:, :h, :w] = data
                data = padded

            # Skip tiles with insufficient data
            if data.shape[0] >= 3:
                # Check first 3 bands for data content
                data_mask = np.any(data[:3] != pad_value, axis=0)
            else:
                data_mask = data[0] != pad_value

            data_fraction = data_mask.sum() / data_mask.size
            if data_fraction < min_data_fraction:
                continue

            # Compute the geotransform for this tile
            tile_transform = rasterio.windows.transform(window, src.transform)

            # Build output profile
            profile = src.profile.copy()
            profile.update(
                {
                    "width": tile_size,
                    "height": tile_size,
                    "transform": tile_transform,
                    "compress": "lzw",
                }
            )

            # Write tile
            stem = src_path.stem
            tile_name = f"{stem}_tile_{idx:04d}.tif"
            tile_path = dst_dir / tile_name

            with rasterio.open(tile_path, "w", **profile) as dst:
                dst.write(data)

            output_paths.append(tile_path)

    logger.info(
        f"Tiled {src_path.name} → {len(output_paths)} tiles "
        f"({tile_size}×{tile_size}, overlap={overlap})"
    )
    return output_paths


def tile_image_and_mask(
    img_path: Union[str, Path],
    mask_path: Union[str, Path],
    img_dst_dir: Union[str, Path],
    mask_dst_dir: Union[str, Path],
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    min_data_fraction: float = 0.1,
) -> list[tuple[Path, Path]]:
    """
    Tile an image and its corresponding mask in sync.

    Ensures that image tiles and mask tiles share the same spatial extent.

    Args:
        img_path: Path to the image GeoTIFF.
        mask_path: Path to the mask GeoTIFF.
        img_dst_dir: Output directory for image tiles.
        mask_dst_dir: Output directory for mask tiles.
        tile_size: Tile size in pixels.
        overlap: Overlap between tiles.
        min_data_fraction: Minimum data fraction to keep.

    Returns:
        List of (image_tile_path, mask_tile_path) tuples.
    """
    img_path = Path(img_path)
    mask_path = Path(mask_path)
    Path(img_dst_dir).mkdir(parents=True, exist_ok=True)
    Path(mask_dst_dir).mkdir(parents=True, exist_ok=True)

    pairs = []

    with rasterio.open(img_path) as img_src, rasterio.open(mask_path) as mask_src:
        windows = compute_tile_windows(
            img_src.width, img_src.height, tile_size, overlap
        )

        for idx, window in enumerate(windows):
            # Read image tile
            img_data = img_src.read(window=window)
            _, h, w = img_data.shape

            # Pad image
            if h < tile_size or w < tile_size:
                padded = np.zeros(
                    (img_data.shape[0], tile_size, tile_size),
                    dtype=img_data.dtype,
                )
                padded[:, :h, :w] = img_data
                img_data = padded

            # Check data content
            if img_data.shape[0] >= 3:
                data_mask = np.any(img_data[:3] != 0, axis=0)
            else:
                data_mask = img_data[0] != 0

            if data_mask.sum() / data_mask.size < min_data_fraction:
                continue

            # Read and pad mask tile
            mask_data = mask_src.read(window=window)
            _, mh, mw = mask_data.shape
            if mh < tile_size or mw < tile_size:
                padded = np.zeros(
                    (mask_data.shape[0], tile_size, tile_size),
                    dtype=mask_data.dtype,
                )
                padded[:, :mh, :mw] = mask_data
                mask_data = padded

            tile_transform = rasterio.windows.transform(
                window, img_src.transform
            )
            stem = img_path.stem

            # Write image tile
            img_profile = img_src.profile.copy()
            img_profile.update(
                width=tile_size,
                height=tile_size,
                transform=tile_transform,
                compress="lzw",
            )
            img_tile_path = Path(img_dst_dir) / f"{stem}_tile_{idx:04d}.tif"
            with rasterio.open(img_tile_path, "w", **img_profile) as dst:
                dst.write(img_data)

            # Write mask tile
            mask_profile = mask_src.profile.copy()
            mask_profile.update(
                width=tile_size,
                height=tile_size,
                transform=tile_transform,
                compress="lzw",
            )
            mask_tile_path = Path(mask_dst_dir) / f"{stem}_tile_{idx:04d}.tif"
            with rasterio.open(mask_tile_path, "w", **mask_profile) as dst:
                dst.write(mask_data)

            pairs.append((img_tile_path, mask_tile_path))

    logger.info(f"Tiled {img_path.name} + mask → {len(pairs)} tile pairs")
    return pairs


def reconstruct_from_tiles(
    tile_paths: list[Union[str, Path]],
    output_path: Union[str, Path],
    original_width: int,
    original_height: int,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> Path:
    """
    Reconstruct a full image from overlapping tiles using averaging.

    Used during inference to stitch predictions back together.

    Args:
        tile_paths: Ordered list of tile file paths.
        output_path: Path for the reconstructed image.
        original_width: Original image width.
        original_height: Original image height.
        tile_size: Tile size used during tiling.
        overlap: Overlap used during tiling.

    Returns:
        Path to the reconstructed image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read first tile to get channel count and dtype
    with rasterio.open(tile_paths[0]) as src:
        num_channels = src.count
        dtype = src.dtypes[0]
        profile = src.profile.copy()

    # Accumulation arrays
    result = np.zeros(
        (num_channels, original_height, original_width), dtype=np.float64
    )
    counts = np.zeros((original_height, original_width), dtype=np.float64)

    stride = tile_size - overlap
    windows = compute_tile_windows(
        original_width, original_height, tile_size, overlap
    )

    for tile_path, window in zip(tile_paths, windows):
        with rasterio.open(tile_path) as src:
            data = src.read().astype(np.float64)

        x, y = int(window.col_off), int(window.row_off)
        h = min(tile_size, original_height - y)
        w = min(tile_size, original_width - x)

        result[:, y : y + h, x : x + w] += data[:, :h, :w]
        counts[y : y + h, x : x + w] += 1.0

    # Average overlapping regions
    counts = np.maximum(counts, 1.0)
    result = result / counts[np.newaxis, :, :]

    # Write output
    profile.update(
        width=original_width,
        height=original_height,
        count=num_channels,
        compress="lzw",
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(result.astype(dtype))

    logger.info(f"Reconstructed {output_path.name} from {len(tile_paths)} tiles")
    return output_path
