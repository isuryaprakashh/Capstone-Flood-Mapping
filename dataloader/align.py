"""
Pre/Post Co-registration for SpaceNet 8 Image Pairs.

SpaceNet 8 tiles the pre-event image on a 1300 x 1300 geographic grid and
cuts the post-event image from the same grid, so both tiles cover the same
ground but at different pixel sizes (post GSD 0.31-0.79 m vs pre 0.37-0.52 m).
This module resamples every post-event tile onto its pre-event pixel grid
(geometric alignment) and measures the residual misalignment that remains
after georeferencing (sensor look-angle, orthorectification error).
"""

import logging
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

logger = logging.getLogger(__name__)


def read_rgb(path: Union[str, Path]) -> tuple[np.ndarray, dict]:
    """
    Read the first three bands of a GeoTIFF as uint8 RGB.

    Args:
        path: Path to the GeoTIFF.

    Returns:
        Tuple of (array (3, H, W) uint8, georeference dict with
        'transform', 'crs', 'res', 'shape').
    """
    with rasterio.open(path) as src:
        data = src.read(list(range(1, min(src.count, 3) + 1)))
        geo = {
            "transform": src.transform,
            "crs": src.crs,
            "res": src.res,
            "shape": (src.height, src.width),
        }

    if data.dtype != np.uint8:
        max_val = float(data.max()) if data.size else 0.0
        scale = 255.0 / (4095.0 if max_val > 255 and max_val <= 4095 else max(max_val, 1.0))
        data = np.clip(data.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)

    return data, geo


def resample_to_grid(
    src_path: Union[str, Path],
    dst_geo: dict,
    resampling: Resampling = Resampling.bilinear,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample a GeoTIFF onto another raster's pixel grid.

    Args:
        src_path: GeoTIFF to resample (the post-event tile).
        dst_geo: Georeference of the target grid (from read_rgb of the pre tile).
        resampling: Resampling kernel.

    Returns:
        Tuple of (resampled array (3, H, W) uint8, valid mask (H, W) bool
        marking pixels covered by the source image).
    """
    src_data, src_geo = read_rgb(src_path)
    height, width = dst_geo["shape"]

    out = np.zeros((src_data.shape[0], height, width), dtype=np.uint8)
    for band in range(src_data.shape[0]):
        reproject(
            source=src_data[band],
            destination=out[band],
            src_transform=src_geo["transform"],
            src_crs=src_geo["crs"],
            dst_transform=dst_geo["transform"],
            dst_crs=dst_geo["crs"],
            resampling=resampling,
            src_nodata=0,
            dst_nodata=0,
        )

    coverage = np.ones(src_geo["shape"], dtype=np.uint8)
    valid = np.zeros((height, width), dtype=np.uint8)
    reproject(
        source=coverage,
        destination=valid,
        src_transform=src_geo["transform"],
        src_crs=src_geo["crs"],
        dst_transform=dst_geo["transform"],
        dst_crs=dst_geo["crs"],
        resampling=Resampling.nearest,
        dst_nodata=0,
    )

    # Black (nodata) pixels inside the source footprint are invalid too
    valid = (valid > 0) & (out.max(axis=0) > 0)
    return out, valid


def load_aligned_pair(
    pre_path: Union[str, Path],
    post_path: Union[str, Path],
) -> dict:
    """
    Load a pre/post pair with the post image resampled onto the pre grid.

    Args:
        pre_path: Pre-event GeoTIFF.
        post_path: Post-event GeoTIFF.

    Returns:
        Dictionary with 'pre' (3, H, W) uint8, 'post' (3, H, W) uint8,
        'valid' (H, W) bool, 'geo' (pre georeference), 'pre_gsd_m',
        'post_gsd_m'.
    """
    pre, pre_geo = read_rgb(pre_path)
    post, valid = resample_to_grid(post_path, pre_geo)

    with rasterio.open(post_path) as src:
        post_res = src.res
        post_crs = src.crs

    return {
        "pre": pre,
        "post": post,
        "valid": valid,
        "geo": pre_geo,
        "pre_gsd_m": pixel_size_m(pre_geo["res"], pre_geo["crs"], pre_geo["transform"]),
        "post_gsd_m": pixel_size_m(post_res, post_crs, pre_geo["transform"]),
    }


def pixel_size_m(res: tuple, crs, transform) -> float:
    """
    Approximate ground pixel size in meters.

    Args:
        res: (x_res, y_res) in CRS units.
        crs: Raster CRS.
        transform: Affine transform (used for latitude of geographic CRSs).

    Returns:
        Mean pixel size in meters.
    """
    if crs is not None and crs.is_geographic:
        lat = np.deg2rad(transform.f)
        meters_per_deg_lat = 111_320.0
        meters_per_deg_lon = 111_320.0 * np.cos(lat)
        return float((abs(res[0]) * meters_per_deg_lon + abs(res[1]) * meters_per_deg_lat) / 2)
    return float((abs(res[0]) + abs(res[1])) / 2)


def estimate_shift(
    pre: np.ndarray,
    post: np.ndarray,
    valid: np.ndarray = None,
) -> dict:
    """
    Estimate the residual global translation between aligned pre/post images.

    Uses phase correlation on gradient-magnitude images, which is robust
    to the illumination and radiometric differences between acquisitions.

    Args:
        pre: Pre image (3, H, W) uint8.
        post: Post image on the same grid (3, H, W) uint8.
        valid: Optional valid-pixel mask (H, W).

    Returns:
        Dictionary with 'dy', 'dx' (post must move by this to match pre,
        in pixels), 'magnitude' (pixels) and 'response' (peak strength,
        higher is more reliable).
    """
    def edges(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(np.ascontiguousarray(img.transpose(1, 2, 0)), cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        if valid is not None:
            mag = mag * valid.astype(np.float32)
        return mag

    a = edges(pre)
    b = edges(post)
    window = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(b, a, window)

    return {
        "dy": float(dy),
        "dx": float(dx),
        "magnitude": float(np.hypot(dx, dy)),
        "response": float(response),
    }


def estimate_shift_blocks(
    pre: np.ndarray,
    post: np.ndarray,
    valid: np.ndarray = None,
    block: int = 325,
    min_response: float = 0.05,
) -> dict:
    """
    Estimate residual misalignment from block-wise phase correlation.

    Whole-tile correlation is dominated by flood-induced appearance change;
    estimating per block and keeping only confident peaks is far more robust
    and also exposes non-rigid (spatially varying) misalignment.

    Args:
        pre: Pre image (3, H, W) uint8.
        post: Post image on the same grid (3, H, W) uint8.
        valid: Optional valid-pixel mask (H, W).
        block: Block size in pixels.
        min_response: Minimum phase-correlation peak to accept a block.

    Returns:
        Dictionary with 'dy', 'dx', 'magnitude' (median over confident
        blocks, NaN if none), 'spread' (median absolute deviation of the
        block shifts, a non-rigidity measure), 'n_blocks', 'n_confident'.
    """
    _, h, w = pre.shape
    shifts = []
    n_blocks = 0
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            v = None if valid is None else valid[y:y + block, x:x + block]
            if v is not None and v.mean() < 0.5:
                continue
            n_blocks += 1
            r = estimate_shift(pre[:, y:y + block, x:x + block], post[:, y:y + block, x:x + block], v)
            if r["response"] >= min_response and r["magnitude"] < block / 4:
                shifts.append((r["dy"], r["dx"]))

    if not shifts:
        nan = float("nan")
        return {"dy": nan, "dx": nan, "magnitude": nan, "spread": nan,
                "n_blocks": n_blocks, "n_confident": 0}

    arr = np.array(shifts)
    med = np.median(arr, axis=0)
    spread = float(np.median(np.hypot(*(arr - med).T)))
    return {
        "dy": float(med[0]),
        "dx": float(med[1]),
        "magnitude": float(np.hypot(*med)),
        "spread": spread,
        "n_blocks": n_blocks,
        "n_confident": len(shifts),
    }


def apply_shift(image: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """
    Translate a (C, H, W) image by (dy, dx) pixels with zero fill.

    Args:
        image: Image array (C, H, W).
        dy: Vertical shift in pixels.
        dx: Horizontal shift in pixels.

    Returns:
        Shifted image, same dtype and shape.
    """
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = image.shape[1:]
    return np.stack(
        [
            cv2.warpAffine(band, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
            for band in image
        ]
    )
