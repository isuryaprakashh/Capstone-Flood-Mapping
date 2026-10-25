"""
Tests for pre/post co-registration utilities and prediction binarization.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from dataloader.align import (
    apply_shift,
    estimate_shift,
    estimate_shift_blocks,
    load_aligned_pair,
)
from inference.predict import binarize

BOUNDS = (-90.0253, 29.8200, -90.0208, 29.8245)


def _smooth_texture(size: int, seed: int = 0) -> np.ndarray:
    import cv2

    rng = np.random.default_rng(seed)
    img = (rng.random((3, size, size)) * 255).astype(np.uint8)
    return np.stack([cv2.GaussianBlur(b, (0, 0), 2) for b in img])


def _write_tif(path, data):
    _, h, w = data.shape
    with rasterio.open(
        path, "w", driver="GTiff", width=w, height=h, count=3, dtype="uint8",
        crs="EPSG:4326", transform=from_bounds(*BOUNDS, w, h),
    ) as dst:
        dst.write(data)


def test_estimate_shift_recovers_known_translation():
    img = _smooth_texture(400)
    post = apply_shift(img, -3.0, 5.0)
    r = estimate_shift(img, post)
    assert r["dy"] == pytest.approx(3.0, abs=0.1)
    assert r["dx"] == pytest.approx(-5.0, abs=0.1)

    fixed = apply_shift(post, r["dy"], r["dx"])
    assert estimate_shift(img, fixed)["magnitude"] < 0.2


def test_block_estimate_matches_global_shift():
    img = _smooth_texture(650, seed=1)
    post = apply_shift(img, 6.0, -9.0)
    r = estimate_shift_blocks(img, post, block=325)
    assert r["n_confident"] >= 2
    assert r["dy"] == pytest.approx(-6.0, abs=0.3)
    assert r["dx"] == pytest.approx(9.0, abs=0.3)


def test_block_estimate_reports_nan_without_signal():
    flat = np.full((3, 650, 650), 128, dtype=np.uint8)
    r = estimate_shift_blocks(flat, flat, block=325)
    assert r["n_confident"] == 0
    assert np.isnan(r["magnitude"])


def test_load_aligned_pair_resamples_post_to_pre_grid(tmp_path):
    pre = _smooth_texture(130, seed=2)
    post = _smooth_texture(79, seed=3)  # same ground, coarser resolution
    _write_tif(tmp_path / "pre.tif", pre)
    _write_tif(tmp_path / "post.tif", post)

    pair = load_aligned_pair(tmp_path / "pre.tif", tmp_path / "post.tif")
    assert pair["pre"].shape == (3, 130, 130)
    assert pair["post"].shape == (3, 130, 130)
    assert pair["valid"].mean() > 0.95
    assert pair["post_gsd_m"] > pair["pre_gsd_m"]


def test_binarize_applies_thresholds_and_overlap_rules():
    probs = np.zeros((5, 4, 4), dtype=np.float32)
    probs[0, :2, :2] = 0.9   # building
    probs[2, :2, :] = 0.9    # flood over top half
    probs[3, 3, 3] = 0.9     # flooded building outside any building
    out = binarize(probs, [0.5, 0.5, 0.5, 0.5, 0.5])

    assert out[3, 0, 0] == 1  # building under flood becomes flooded building
    assert out[3, 3, 3] == 0  # flooded building must lie on a building
    assert out[4].sum() == 0  # no roads → no flooded roads
