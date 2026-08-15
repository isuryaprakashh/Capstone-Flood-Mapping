"""
Inference-time evaluation utilities.

Provides helpers for computing metrics on inference outputs,
generating visualizations, and comparing predictions.
"""

import numpy as np

CLASS_NAMES = ["building", "road", "flood", "flooded_building", "flooded_road"]


def compute_prediction_metrics(
    prediction: dict, ground_truth: np.ndarray
) -> dict:
    """
    Compute metrics between prediction output and ground truth mask.

    Args:
        prediction: Output from FloodPredictor.predict().
        ground_truth: Ground truth mask (NUM_CLASSES, H, W).

    Returns:
        Dictionary of per-class metrics.
    """
    binary = prediction["binary_mask"]
    metrics = {}

    for c, name in enumerate(CLASS_NAMES):
        if c >= binary.shape[0] or c >= ground_truth.shape[0]:
            break

        pred_c = binary[c].flatten().astype(float)
        targ_c = ground_truth[c].flatten().astype(float)

        tp = (pred_c * targ_c).sum()
        fp = (pred_c * (1 - targ_c)).sum()
        fn = ((1 - pred_c) * targ_c).sum()

        metrics[f"iou_{name}"] = float(tp / (tp + fp + fn + 1e-8))
        metrics[f"f1_{name}"] = float(2 * tp / (2 * tp + fp + fn + 1e-8))

    ious = [metrics[f"iou_{n}"] for n in CLASS_NAMES if f"iou_{n}" in metrics]
    metrics["mean_iou"] = float(np.mean(ious)) if ious else 0.0

    return metrics


def mask_to_rgba(
    binary_mask: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Convert multi-class binary mask to an RGBA overlay image.

    Color scheme:
      - Building: Blue
      - Road: White
      - Flood: Cyan
      - Flooded building: Red
      - Flooded road: Orange

    Args:
        binary_mask: Binary mask (NUM_CLASSES, H, W).
        alpha: Overlay alpha value.

    Returns:
        RGBA image (H, W, 4) with uint8 values.
    """
    _, H, W = binary_mask.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)

    # Color map: (R, G, B)
    colors = {
        0: (66, 133, 244),    # Building: Blue
        1: (220, 220, 220),   # Road: Light gray
        2: (0, 200, 255),     # Flood: Cyan
        3: (234, 67, 53),     # Flooded building: Red
        4: (251, 188, 4),     # Flooded road: Orange
    }

    # Render from bottom to top priority
    for c in [1, 0, 2, 4, 3]:
        if c >= binary_mask.shape[0]:
            continue
        mask = binary_mask[c] > 0
        if mask.any():
            r, g, b = colors[c]
            rgba[mask, 0] = r
            rgba[mask, 1] = g
            rgba[mask, 2] = b
            rgba[mask, 3] = int(alpha * 255)

    return rgba
