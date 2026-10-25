"""
Tests for building matching and flood-attribution metrics (evaluation.buildings).
"""

import numpy as np
import pytest

from evaluation.buildings import (
    aggregate_building_counts,
    building_metrics,
    match_buildings,
)


def _rect(x0, y0, x1, y1, flooded=False):
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
        },
        "properties": {"flooded": flooded},
    }


def _fc(*features):
    return {"type": "FeatureCollection", "features": list(features)}


# ─── Matching ───────────────────────────────────────────────────────────────


def test_identical_buildings_match_with_iou_one():
    fc = _fc(_rect(0, 0, 10, 10), _rect(20, 20, 30, 35))
    result = match_buildings(fc, fc)
    assert [(p, g) for p, g, _ in result["matches"]] == [(0, 0), (1, 1)]
    assert all(iou == pytest.approx(1.0) for _, _, iou in result["matches"])
    assert result["unmatched_pred"] == [] and result["unmatched_gt"] == []


def test_iou_exactly_at_threshold_matches():
    gt = _fc(_rect(0, 0, 10, 10))
    pred = _fc(_rect(0, 0, 10, 20))  # IoU = 100 / 200 = 0.5
    result = match_buildings(pred, gt, iou_threshold=0.5)
    assert len(result["matches"]) == 1
    assert result["matches"][0][2] == pytest.approx(0.5)


def test_iou_just_below_threshold_does_not_match():
    gt = _fc(_rect(0, 0, 10, 10))
    pred = _fc(_rect(0, 0, 10, 20.5))  # IoU = 100 / 205 ~ 0.488
    result = match_buildings(pred, gt, iou_threshold=0.5)
    assert result["matches"] == []
    assert result["unmatched_pred"] == [0] and result["unmatched_gt"] == [0]
    assert len(match_buildings(pred, gt, iou_threshold=0.45)["matches"]) == 1


def test_touching_buildings_never_match_even_at_zero_threshold():
    result = match_buildings(_fc(_rect(10, 0, 20, 10)), _fc(_rect(0, 0, 10, 10)), iou_threshold=0.0)
    assert result["matches"] == []


def test_matching_is_one_to_one_and_prefers_higher_iou():
    gt = _fc(_rect(0, 0, 10, 10))
    pred = _fc(_rect(0, 0, 10, 15), _rect(0, 0, 10, 11))  # IoU 0.667 and 0.909
    result = match_buildings(pred, gt)
    assert [(p, g) for p, g, _ in result["matches"]] == [(1, 0)]
    assert result["unmatched_pred"] == [0]


def test_matching_reports_original_indices_when_skipping_empty_geometry():
    empty = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": []}, "properties": {}}
    gt = _fc(_rect(0, 0, 10, 10))
    pred = _fc(empty, _rect(0, 0, 10, 10))
    result = match_buildings(pred, gt)
    assert [(p, g) for p, g, _ in result["matches"]] == [(1, 0)]
    assert result["unmatched_pred"] == []


def test_invalid_polygon_is_repaired():
    bowtie = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [10, 10], [10, 0], [0, 10], [0, 0]]]},
        "properties": {"flooded": False},
    }
    result = match_buildings(_fc(bowtie), _fc(_rect(0, 0, 10, 10)))
    # Repaired into two triangles of 25 px2 each, inside the square: IoU 0.5
    assert len(result["matches"]) == 1
    assert result["matches"][0][2] == pytest.approx(0.5)


# ─── Metrics ────────────────────────────────────────────────────────────────


def test_empty_inputs():
    result = building_metrics(_fc(), _fc(), tile_shape=(32, 32))
    m = result["metrics"]
    assert result["counts"]["all"] == {"tp": 0, "fp": 0, "fn": 0}
    assert m["all"]["f1"] is None and m["all"]["precision"] is None and m["all"]["recall"] is None
    assert m["flood_iou"] is None

    only_pred = building_metrics(_fc(_rect(0, 0, 5, 5)), _fc())["metrics"]["all"]
    assert only_pred["precision"] == 0.0 and only_pred["recall"] is None and only_pred["f1"] == 0.0


def test_flood_state_f1_requires_match_and_correct_state():
    gt = _fc(
        _rect(0, 0, 10, 10, True),      # A: found, correct state
        _rect(20, 0, 30, 10, True),     # B: found, predicted non-flooded
        _rect(40, 0, 50, 10, False),    # C: found, correct state
        _rect(60, 0, 70, 10, False),    # D: missed
    )
    pred = _fc(
        _rect(0, 0, 10, 10, True),
        _rect(20, 0, 30, 10, False),
        _rect(40, 0, 50, 10, False),
        _rect(80, 0, 90, 10, True),     # E: spurious
    )
    result = building_metrics(pred, gt)
    c, m = result["counts"], result["metrics"]
    assert c["all"] == {"tp": 3, "fp": 1, "fn": 1}
    assert c["flooded"] == {"tp": 1, "fp": 1, "fn": 1}
    assert c["non_flooded"] == {"tp": 1, "fp": 1, "fn": 1}
    assert c["state_correct"] == 2
    assert m["all"]["f1"] == pytest.approx(0.75)
    assert m["flooded"]["f1"] == pytest.approx(0.5)
    assert m["non_flooded"]["f1"] == pytest.approx(0.5)
    assert m["state_accuracy"] == pytest.approx(2 / 3)
    assert "pixel" not in c and "flood_iou" not in m


def test_pixel_flood_iou_perfect_and_flipped():
    gt = _fc(_rect(10, 10, 30, 30, True), _rect(50, 50, 70, 60, False))
    perfect = building_metrics(gt, gt, tile_shape=(100, 100))
    assert perfect["counts"]["pixel"]["flooded"] == {"tp": 400, "fp": 0, "fn": 0}
    assert perfect["counts"]["pixel"]["non_flooded"] == {"tp": 200, "fp": 0, "fn": 0}
    assert perfect["metrics"]["flood_iou"] == pytest.approx(1.0)

    flipped = _fc(_rect(10, 10, 30, 30, False), _rect(50, 50, 70, 60, True))
    m = building_metrics(flipped, gt, tile_shape=(100, 100))["metrics"]
    assert m["pixel_iou_flooded"] == 0.0 and m["pixel_iou_non_flooded"] == 0.0
    assert m["flood_iou"] == 0.0
    assert m["all"]["f1"] == pytest.approx(1.0)
    assert m["flooded"]["f1"] == 0.0


def test_pixel_flood_iou_absent_class_is_none():
    gt = _fc(_rect(10, 10, 30, 30, False))
    pred = _fc(_rect(10, 10, 30, 20, False))
    m = building_metrics(pred, gt, tile_shape=(64, 64))["metrics"]
    assert m["pixel_iou_flooded"] is None
    assert m["pixel_iou_non_flooded"] == pytest.approx(0.5)
    assert m["flood_iou"] == pytest.approx(0.5)


def test_valid_mask_restricts_pixel_counts():
    gt = _fc(_rect(0, 0, 20, 10, True))
    valid = np.zeros((32, 32), dtype=bool)
    valid[:, :10] = True
    counts = building_metrics(_fc(), gt, valid=valid)["counts"]["pixel"]["flooded"]
    assert counts == {"tp": 0, "fp": 0, "fn": 100}
    with pytest.raises(ValueError):
        building_metrics(_fc(), gt, tile_shape=(16, 16), valid=valid)


def test_aggregate_micro_averages_counts():
    gt1 = _fc(_rect(0, 0, 10, 10, True), _rect(20, 0, 30, 10))
    pred1 = _fc(_rect(0, 0, 10, 10, True))
    gt2 = _fc(_rect(0, 0, 10, 10))
    pred2 = _fc(_rect(0, 0, 10, 10), _rect(50, 50, 60, 60), _rect(70, 70, 80, 80))
    r1 = building_metrics(pred1, gt1, tile_shape=(100, 100))
    r2 = building_metrics(pred2, gt2, tile_shape=(100, 100))
    agg = aggregate_building_counts([r1, r2])
    assert agg["n_tiles"] == 2
    assert agg["counts"]["all"] == {"tp": 2, "fp": 2, "fn": 1}
    assert agg["metrics"]["all"]["f1"] == pytest.approx(4 / 7)
    assert agg["counts"]["pixel"]["flooded"]["tp"] == 100
    assert aggregate_building_counts([r1["counts"], r2["counts"]])["counts"] == agg["counts"]

    no_pixel = building_metrics(pred2, gt2)
    with pytest.raises(ValueError):
        aggregate_building_counts([r1, no_pixel])
    assert aggregate_building_counts([])["metrics"]["all"]["f1"] is None
