"""
Length-Based Road Precision, Recall and F1.

A complement to APLS that ignores topology: the share of reference road
length lying within a buffer of the predicted centerlines (recall) and the
share of predicted length within the same buffer of the reference
(precision). Overlapping or duplicated lines are dissolved first so no
length is counted twice. Lengths are returned in meters so tiles can be
aggregated by summing them (aggregate_road_lengths).
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Optional

import shapely
from shapely.geometry import shape

from evaluation.common import safe_div, sum_counts

logger = logging.getLogger(__name__)

LENGTH_KEYS = ("gt_m", "pred_m", "gt_matched_m", "pred_matched_m")


def _dissolved_lines(fc: Mapping, flooded: Optional[bool] = None) -> shapely.Geometry:
    """Union of the line features, optionally filtered by flood state."""
    lines = []
    for feature in fc.get("features", []):
        if flooded is not None and bool((feature.get("properties") or {}).get("flooded", False)) != flooded:
            continue
        geom = shape(feature["geometry"])
        if geom.is_empty or geom.geom_type not in ("LineString", "MultiLineString"):
            continue
        lines.append(geom)
    if not lines:
        return shapely.LineString()
    return shapely.unary_union(lines)


def length_f1_from_counts(lengths: Mapping) -> dict:
    """
    Precision, recall and F1 from (possibly aggregated) road lengths.

    Precision is None without predicted length and recall None without
    reference length; F1 (harmonic mean of the two, an undefined one counted
    as 0) is None only when both are absent.

    Args:
        lengths: Dict with 'gt_m', 'pred_m', 'gt_matched_m', 'pred_matched_m'.

    Returns:
        Dict with 'precision', 'recall' and 'f1'.
    """
    precision = safe_div(lengths["pred_matched_m"], lengths["pred_m"])
    recall = safe_div(lengths["gt_matched_m"], lengths["gt_m"])
    if precision is None and recall is None:
        f1 = None
    else:
        p, r = precision or 0.0, recall or 0.0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def road_length_f1(
    pred_roads_fc: Mapping,
    gt_roads_fc: Mapping,
    gsd_m: float,
    buffer_m: float = 4.0,
    flooded: Optional[bool] = None,
) -> dict:
    """
    Buffer-based length precision / recall / F1 of road centerlines.

    Args:
        pred_roads_fc: Predicted road FeatureCollection (pixel coordinates).
        gt_roads_fc: Reference road FeatureCollection (pixel coordinates).
        gsd_m: Meters per pixel.
        buffer_m: Matching tolerance around each centerline, in meters.
        flooded: If True or False, keep only features with that flood state
            on both sides; None uses all roads.

    Returns:
        Dict with lengths in meters ('gt_m', 'pred_m', 'gt_matched_m',
        'pred_matched_m') and 'precision', 'recall', 'f1'.
    """
    if gsd_m <= 0:
        raise ValueError(f"gsd_m must be positive, got {gsd_m}")
    buffer_px = buffer_m / gsd_m
    pred = _dissolved_lines(pred_roads_fc, flooded)
    gt = _dissolved_lines(gt_roads_fc, flooded)

    gt_matched = pred_matched = 0.0
    if not pred.is_empty and not gt.is_empty:
        gt_matched = gt.intersection(pred.buffer(buffer_px)).length
        pred_matched = pred.intersection(gt.buffer(buffer_px)).length

    lengths = {
        "gt_m": gt.length * gsd_m,
        "pred_m": pred.length * gsd_m,
        "gt_matched_m": gt_matched * gsd_m,
        "pred_matched_m": pred_matched * gsd_m,
    }
    return {**lengths, **length_f1_from_counts(lengths)}


def flooded_road_f1(
    pred_roads_fc: Mapping,
    gt_roads_fc: Mapping,
    gsd_m: float,
    buffer_m: float = 4.0,
) -> dict:
    """
    Length-based precision / recall / F1 of flooded roads.

    Recall is the reference flooded length within buffer_m of predicted
    flooded centerlines over the total reference flooded length; precision
    is the predicted flooded length within buffer_m of reference flooded
    centerlines over the total predicted flooded length.

    Args:
        pred_roads_fc: Predicted road FeatureCollection with properties.flooded.
        gt_roads_fc: Reference road FeatureCollection with properties.flooded.
        gsd_m: Meters per pixel.
        buffer_m: Matching tolerance in meters.

    Returns:
        Same structure as road_length_f1().
    """
    return road_length_f1(pred_roads_fc, gt_roads_fc, gsd_m, buffer_m=buffer_m, flooded=True)


def aggregate_road_lengths(results: Sequence[Mapping]) -> dict:
    """
    Micro-average length-based road metrics over tiles.

    Args:
        results: Outputs of road_length_f1() or flooded_road_f1().

    Returns:
        Dict with summed lengths, recomputed 'precision', 'recall', 'f1' and
        'n_tiles'.
    """
    lengths = sum_counts([{k: r[k] for k in LENGTH_KEYS} for r in results])
    lengths = {k: float(lengths.get(k, 0.0)) for k in LENGTH_KEYS}
    return {**lengths, **length_f1_from_counts(lengths), "n_tiles": len(results)}
