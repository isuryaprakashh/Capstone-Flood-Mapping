"""
Shared Count Arithmetic for Evaluation Metrics.

Every metric in this package is computed from additive counts (pixels,
objects, meters), so per-tile results can be summed and the ratio metrics
recomputed on the totals (micro-averaging). Ratios with a zero denominator
are undefined and reported as None rather than 0 or 1.
"""

import logging
from collections.abc import Iterable, Mapping
from typing import Optional

logger = logging.getLogger(__name__)


def safe_div(num: float, den: float) -> Optional[float]:
    """
    Divide, returning None when the denominator is zero.

    Args:
        num: Numerator.
        den: Denominator.

    Returns:
        num / den as a float, or None when den == 0.
    """
    return float(num) / float(den) if den else None


def prf_from_counts(tp: float, fp: float, fn: float) -> dict:
    """
    Precision, recall, F1 and IoU from true/false positive/negative counts.

    Precision is undefined without predictions (tp + fp == 0), recall without
    references (tp + fn == 0), and F1/IoU when both are absent
    (tp + fp + fn == 0); undefined values are None.

    Args:
        tp: True positives.
        fp: False positives.
        fn: False negatives.

    Returns:
        Dict with 'precision', 'recall', 'f1' and 'iou'.
    """
    return {
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "iou": safe_div(tp, tp + fp + fn),
    }


def sum_counts(items: Iterable[Mapping]) -> dict:
    """
    Sum (possibly nested) count dictionaries key by key.

    Numeric leaves are added; nested mappings are summed recursively; keys
    missing from some items count as zero.

    Args:
        items: Count dictionaries with identical or overlapping structure.

    Returns:
        A new dictionary with the summed counts.
    """
    total: dict = {}
    for item in items:
        for key, value in item.items():
            if isinstance(value, Mapping):
                total[key] = sum_counts([total.get(key, {}), value])
            elif isinstance(value, (bool, type(None))):
                raise TypeError(f"Count '{key}' must be numeric, got {value!r}")
            else:
                total[key] = total.get(key, 0) + value
    return total


def mean_defined(values: Iterable[Optional[float]]) -> Optional[float]:
    """
    Mean of the values that are not None.

    Args:
        values: Numbers, some possibly None.

    Returns:
        The mean of the defined values, or None if there are none.
    """
    defined = [float(v) for v in values if v is not None]
    return sum(defined) / len(defined) if defined else None
