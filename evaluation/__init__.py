"""
Evaluation Suite for SpaceNet 8 Flood Mapping.

Object-level building metrics with flood attribution, road APLS (including
the SpaceNet 8 traversable / non-traversable variant), length-based road
F1, pixel metrics, tile-level bootstrap confidence intervals and helpers for
the misalignment stress test. Every metric exposes additive counts so tiles
can be micro-averaged.
"""

from evaluation.apls import (
    apls,
    apls_detailed,
    apls_flood,
    build_gt_graph,
    flood_subgraph,
    prediction_graph_from_postprocess,
)
from evaluation.bootstrap import bootstrap_ci
from evaluation.buildings import (
    aggregate_building_counts,
    building_metrics,
    building_metrics_from_counts,
    match_buildings,
)
from evaluation.common import mean_defined, prf_from_counts, safe_div, sum_counts
from evaluation.pixels import class_pixel_counts, pixel_counts, pixel_metrics
from evaluation.roads import (
    aggregate_road_lengths,
    flooded_road_f1,
    length_f1_from_counts,
    road_length_f1,
)
from evaluation.stress import shift_image, shift_schedule

__all__ = [
    "aggregate_building_counts",
    "aggregate_road_lengths",
    "apls",
    "apls_detailed",
    "apls_flood",
    "bootstrap_ci",
    "build_gt_graph",
    "building_metrics",
    "building_metrics_from_counts",
    "class_pixel_counts",
    "flood_subgraph",
    "flooded_road_f1",
    "length_f1_from_counts",
    "match_buildings",
    "mean_defined",
    "pixel_counts",
    "pixel_metrics",
    "prediction_graph_from_postprocess",
    "prf_from_counts",
    "road_length_f1",
    "safe_div",
    "shift_image",
    "shift_schedule",
    "sum_counts",
]
