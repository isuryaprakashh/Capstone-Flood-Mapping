"""
CRS Reprojection Utilities for SpaceNet 8 Satellite Imagery.

Handles coordinate reference system alignment between pre-event and
post-event GeoTIFF imagery, ensuring spatial consistency for model input.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import rasterio
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
)

logger = logging.getLogger(__name__)

# Default target CRS — WGS84 geographic
DEFAULT_TARGET_CRS = "EPSG:4326"


def get_crs_info(filepath: Union[str, Path]) -> dict:
    """
    Extract CRS metadata from a GeoTIFF file.

    Args:
        filepath: Path to the GeoTIFF file.

    Returns:
        Dictionary with CRS info: epsg code, wkt string, bounds, resolution.
    """
    with rasterio.open(filepath) as src:
        return {
            "crs": src.crs,
            "epsg": src.crs.to_epsg() if src.crs else None,
            "bounds": src.bounds,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "res": src.res,
            "count": src.count,
            "dtype": src.dtypes[0],
        }


def reproject_raster(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    target_crs: str = DEFAULT_TARGET_CRS,
    resampling: Resampling = Resampling.bilinear,
    resolution: Optional[float] = None,
) -> Path:
    """
    Reproject a GeoTIFF to a target CRS.

    Args:
        src_path: Path to source GeoTIFF.
        dst_path: Path to write reprojected GeoTIFF.
        target_crs: Target CRS string (default: EPSG:4326).
        resampling: Resampling method for reprojection.
        resolution: Optional target resolution in CRS units.

    Returns:
        Path to the reprojected file.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        # Check if already in target CRS
        if src.crs and src.crs.to_string() == target_crs:
            logger.debug(f"File already in {target_crs}: {src_path.name}")
            # Copy as-is if no resolution change needed
            if resolution is None:
                import shutil
                shutil.copy2(src_path, dst_path)
                return dst_path

        # Calculate the transform for the target CRS
        transform, width, height = calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=resolution,
        )

        # Prepare output profile
        profile = src.profile.copy()
        profile.update(
            {
                "crs": target_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "compress": "lzw",
            }
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=resampling,
                )

    logger.info(f"Reprojected {src_path.name} → {target_crs}")
    return dst_path


def check_crs_alignment(
    paths: list[Union[str, Path]],
    target_crs: str = DEFAULT_TARGET_CRS,
) -> dict:
    """
    Check if a list of GeoTIFF files share the same CRS.

    Args:
        paths: List of GeoTIFF file paths.
        target_crs: Expected CRS to check against.

    Returns:
        Dictionary with alignment status and details.
    """
    results = {"aligned": True, "target_crs": target_crs, "files": []}

    for path in paths:
        info = get_crs_info(path)
        crs_str = info["crs"].to_string() if info["crs"] else "NONE"
        is_match = crs_str == target_crs

        results["files"].append(
            {
                "path": str(path),
                "crs": crs_str,
                "matches_target": is_match,
            }
        )

        if not is_match:
            results["aligned"] = False

    return results


def batch_reproject(
    src_dir: Union[str, Path],
    dst_dir: Union[str, Path],
    target_crs: str = DEFAULT_TARGET_CRS,
    pattern: str = "*.tif",
    resampling: Resampling = Resampling.bilinear,
) -> list[Path]:
    """
    Batch reproject all GeoTIFFs in a directory.

    Args:
        src_dir: Source directory containing GeoTIFFs.
        dst_dir: Output directory for reprojected files.
        target_crs: Target CRS.
        pattern: Glob pattern for file matching.
        resampling: Resampling method.

    Returns:
        List of paths to reprojected files.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_files = sorted(src_dir.glob(pattern))
    output_paths = []

    for src_file in src_files:
        dst_file = dst_dir / src_file.name
        try:
            out = reproject_raster(src_file, dst_file, target_crs, resampling)
            output_paths.append(out)
        except Exception as e:
            logger.error(f"Failed to reproject {src_file.name}: {e}")

    logger.info(
        f"Batch reprojected {len(output_paths)}/{len(src_files)} files "
        f"from {src_dir} → {dst_dir}"
    )
    return output_paths
