"""
Tile-Level Bootstrap Confidence Intervals.

Tiles are the independent sampling unit: each bootstrap replicate resamples
tiles with replacement and recomputes the statistic on the replicate, so
ratio metrics are handled correctly by resampling per-tile count dicts and
re-aggregating them (micro-average) rather than averaging per-tile ratios.
"""

import logging
from collections.abc import Callable, Sequence
from typing import Any, Optional

import numpy as np

from evaluation.common import mean_defined

logger = logging.getLogger(__name__)


def bootstrap_ci(
    per_tile_values: Sequence[Any],
    stat_fn: Optional[Callable[[list], Optional[float]]] = None,
    n: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """
    Percentile bootstrap confidence interval over tiles.

    Examples of stat_fn:
        mean of per-tile APLS: the default (None values are ignored);
        micro-averaged F1: lambda tiles: aggregate_building_counts(tiles)
        ['metrics']['all']['f1'], with per_tile_values the building_metrics()
        results;
        paired model comparison: per_tile_values = list of (a, b) tuples and
        stat_fn = lambda tiles: mean(a) - mean(b) over the replicate.

    Args:
        per_tile_values: One entry per tile (number, count dict, tuple...).
        stat_fn: Maps a list of per-tile entries to a scalar (or None when
            undefined). Defaults to the mean of the non-None numbers.
        n: Number of bootstrap replicates.
        seed: Random seed; identical inputs and seed give identical output.
        alpha: Two-sided miscoverage; the interval spans the alpha/2 and
            1 - alpha/2 percentiles of the replicate statistics.

    Returns:
        Dict with 'estimate' (statistic on all tiles), 'low', 'high', 'std'
        (of the replicate statistics), 'n_boot' (replicates with a defined
        statistic), 'n_tiles' and 'alpha'. Interval fields are None when no
        replicate is defined.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    stat_fn = stat_fn or mean_defined
    values = list(per_tile_values)
    result = {
        "estimate": None,
        "low": None,
        "high": None,
        "std": None,
        "n_boot": 0,
        "n_tiles": len(values),
        "alpha": alpha,
    }
    if not values:
        return result

    estimate = stat_fn(values)
    result["estimate"] = None if estimate is None else float(estimate)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n, len(values)))
    stats = []
    for row in indices:
        s = stat_fn([values[i] for i in row])
        if s is not None and np.isfinite(s):
            stats.append(float(s))
    if len(stats) < n:
        logger.warning(f"Statistic undefined in {n - len(stats)} of {n} bootstrap replicates")
    if not stats:
        return result

    stats = np.asarray(stats)
    low, high = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    result.update(low=float(low), high=float(high), std=float(stats.std(ddof=1)) if len(stats) > 1 else 0.0)
    result["n_boot"] = len(stats)
    return result
