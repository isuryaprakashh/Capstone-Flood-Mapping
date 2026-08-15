"""
Tests for standalone prediction stub module.
"""

from inference.predict import create_stub_prediction
from inference.eval import compute_prediction_metrics, mask_to_rgba
import numpy as np


def test_create_stub_prediction_shapes():
    """Verify stub prediction returns 5 classes with correct dimensions."""
    pred = create_stub_prediction(height=256, width=256)
    assert "probabilities" in pred
    assert "binary_mask" in pred
    assert "class_names" in pred

    assert pred["probabilities"].shape == (5, 256, 256)
    assert pred["binary_mask"].shape == (5, 256, 256)
    assert len(pred["class_names"]) == 5


def test_mask_to_rgba_overlay():
    """Verify mask_to_rgba converts 5-channel mask to 4-channel uint8 RGBA."""
    pred = create_stub_prediction(height=64, width=64)
    rgba = mask_to_rgba(pred["binary_mask"], alpha=0.6)

    assert rgba.shape == (64, 64, 4)
    assert rgba.dtype == np.uint8
    assert rgba[:, :, 3].max() > 0  # has alpha values


def test_prediction_metrics_computation():
    """Verify prediction metric calculations."""
    pred = create_stub_prediction(height=64, width=64)
    gt = pred["binary_mask"].copy()

    metrics = compute_prediction_metrics(pred, gt)
    assert "mean_iou" in metrics
    assert metrics["mean_iou"] == 1.0  # Identity match
