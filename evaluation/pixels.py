"""
Pixel-Level Segmentation Metrics.

Accumulates per-class true positive / false positive / false negative pixel
counts, optionally restricted to a valid-data mask, and derives IoU, F1,
precision and recall from the counts. Counts from many tiles can be summed
with evaluation.common.sum_counts before computing metrics (micro-average).
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Optional

import numpy as np

from evaluation.common import prf_from_counts

logger = logging.getLogger(__name__)


def pixel_counts(
    pred: np.ndarray,
    target: np.ndarray,
    valid: Optional[np.ndarray] = None,
) -> dict:
    """
    Count true positive, false positive and false negative pixels.

    Args:
        pred: Binary prediction (any shape; nonzero = positive).
        target: Binary reference of the same shape.
        valid: Optional boolean mask of the same shape; pixels outside it
            are ignored.

    Returns:
        Dict with integer 'tp', 'fp' and 'fn'.
    """
    pred = np.asarray(pred) > 0
    target = np.asarray(target) > 0
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != pred.shape:
            raise ValueError(f"Shape mismatch: valid {valid.shape} vs pred {pred.shape}")
        pred = pred & valid
        target = target & valid
    tp = int(np.count_nonzero(pred & target))
    return {
        "tp": tp,
        "fp": int(np.count_nonzero(pred)) - tp,
        "fn": int(np.count_nonzero(target)) - tp,
    }


def class_pixel_counts(
    pred: np.ndarray,
    target: np.ndarray,
    class_names: Sequence[str],
    valid: Optional[np.ndarray] = None,
) -> dict:
    """
    Per-class pixel counts for multi-channel binary masks.

    Args:
        pred: Binary prediction (C, H, W).
        target: Binary reference (C, H, W).
        class_names: Name of each of the first len(class_names) channels.
        valid: Optional boolean (H, W) mask shared by all channels.

    Returns:
        Dict mapping class name to {'tp', 'fp', 'fn'}.
    """
    if pred.shape[0] < len(class_names) or target.shape[0] < len(class_names):
        raise ValueError(
            f"Need {len(class_names)} channels, got pred {pred.shape[0]} and target {target.shape[0]}"
        )
    return {
        name: pixel_counts(pred[c], target[c], valid)
        for c, name in enumerate(class_names)
    }


def pixel_metrics(counts: Mapping) -> dict:
    """
    IoU, F1, precision and recall from pixel counts.

    Args:
        counts: Either a single {'tp', 'fp', 'fn'} dict or a dict mapping
            class name to such dicts.

    Returns:
        For a single count dict: {'iou', 'f1', 'precision', 'recall'}, each
        None when undefined (IoU and F1 when tp + fp + fn == 0, precision
        when tp + fp == 0, recall when tp + fn == 0). For per-class counts:
        a dict mapping class name to those metrics.
    """
    if "tp" in counts:
        return prf_from_counts(counts["tp"], counts.get("fp", 0), counts.get("fn", 0))
    return {name: pixel_metrics(c) for name, c in counts.items()}
