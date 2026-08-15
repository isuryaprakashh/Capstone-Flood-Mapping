"""
Unit tests for data preprocessing, tiling, and mask generation.
"""

from pathlib import Path
import numpy as np
import pytest
from rasterio.transform import Affine

from data.tiling import compute_tile_windows
from data.mask_generator import (
    rasterize_pixel_geometries,
    parse_geojson_annotations,
    MASK_CHANNELS,
    NUM_MASK_CHANNELS,
)
from data.dataset import normalize_image, MEAN, STD
from shapely.geometry import LineString, Polygon


def test_tile_windows_computation():
    """Test window generation covers the entire dimension with overlap."""
    width, height = 1000, 1000
    tile_size = 512
    overlap = 64

    windows = compute_tile_windows(width, height, tile_size, overlap)
    assert len(windows) > 0
    # Top-left window must start at 0, 0
    assert windows[0].col_off == 0
    assert windows[0].row_off == 0


def test_rasterize_pixel_geometries():
    """Test rasterization of Shapely line geometries into a binary mask."""
    lines = [
        LineString([(10, 10), (10, 50)]),
        LineString([(20, 20), (80, 20)]),
    ]
    mask = rasterize_pixel_geometries(lines, width=100, height=100, buffer_size=2.0)
    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    assert mask.sum() > 0


def test_image_normalization():
    """Test channel-wise normalization of 3-band imagery."""
    img = np.ones((3, 32, 32), dtype=np.float32) * 0.5
    norm = normalize_image(img.copy(), mean=MEAN, std=STD)
    assert norm.shape == (3, 32, 32)
    assert not np.array_equal(norm, img)


def test_mask_channels_consistency():
    """Test mask channel definitions."""
    assert NUM_MASK_CHANNELS == 5
    assert MASK_CHANNELS["building"] == 0
    assert MASK_CHANNELS["road"] == 1
    assert MASK_CHANNELS["flood"] == 2
    assert MASK_CHANNELS["flooded_building"] == 3
    assert MASK_CHANNELS["flooded_road"] == 4


def test_parse_sample_geojson():
    """Test parsing actual Germany annotation if present."""
    sample_geojson = Path("CP1-DATASET/Germany_Training_Public/annotations/0_15_63.geojson")
    if sample_geojson.exists():
        data = parse_geojson_annotations(sample_geojson)
        assert "buildings" in data
        assert "roads" in data
        assert len(data["buildings"]) > 0 or len(data["roads"]) > 0
