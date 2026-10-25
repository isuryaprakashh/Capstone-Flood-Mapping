"""
Tests for length-based road precision / recall / F1 (evaluation.roads).
"""

import pytest

from evaluation.roads import aggregate_road_lengths, flooded_road_f1, road_length_f1

GSD = 0.5


def _fc(lines, flooded=None):
    flooded = flooded or [False] * len(lines)
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [list(map(float, p)) for p in line]},
                "properties": {"flooded": f},
            }
            for line, f in zip(lines, flooded)
        ],
    }


ROAD = [[(0, 50), (100, 50)], [(100, 50), (200, 50)]]


def test_identical_roads():
    r = road_length_f1(_fc(ROAD), _fc(ROAD), GSD)
    assert r["gt_m"] == pytest.approx(100.0)
    assert r["pred_m"] == pytest.approx(100.0)
    assert r["precision"] == pytest.approx(1.0)
    assert r["recall"] == pytest.approx(1.0)
    assert r["f1"] == pytest.approx(1.0)


def test_offset_within_and_beyond_buffer():
    near = [[(x, y + 6) for x, y in seg] for seg in ROAD]   # 3 m < 4 m
    far = [[(x, y + 20) for x, y in seg] for seg in ROAD]   # 10 m > 4 m
    assert road_length_f1(_fc(near), _fc(ROAD), GSD)["f1"] == pytest.approx(1.0)
    r = road_length_f1(_fc(far), _fc(ROAD), GSD)
    assert r["precision"] == 0.0 and r["recall"] == 0.0 and r["f1"] == 0.0


def test_partial_prediction():
    r = road_length_f1(_fc(ROAD[:1]), _fc(ROAD), GSD)
    assert r["recall"] == pytest.approx(0.5, abs=0.05)  # buffer adds ~4 m of recall
    assert r["precision"] == pytest.approx(1.0)
    assert r["f1"] == pytest.approx(2 * r["recall"] / (1 + r["recall"]))


def test_duplicate_reference_lines_are_not_double_counted():
    r = road_length_f1(_fc(ROAD), _fc(ROAD + ROAD[:1]), GSD)
    assert r["gt_m"] == pytest.approx(100.0)
    assert r["recall"] == pytest.approx(1.0)


def test_flooded_road_f1_uses_flooded_features_only():
    gt = _fc(ROAD + [[(0, 300), (200, 300)]], [True, True, False])
    pred = _fc(ROAD + [[(0, 300), (200, 300)]], [True, False, True])
    r = flooded_road_f1(pred, gt, GSD)
    assert r["gt_m"] == pytest.approx(100.0)
    assert r["pred_m"] == pytest.approx(150.0)
    assert r["recall"] == pytest.approx(0.5, abs=0.05)
    assert r["precision"] == pytest.approx(50.0 / 150.0, abs=0.05)
    assert road_length_f1(pred, gt, GSD)["f1"] == pytest.approx(1.0)


def test_empty_cases():
    both = flooded_road_f1(_fc([]), _fc([]), GSD)
    assert both["precision"] is None and both["recall"] is None and both["f1"] is None
    only_pred = flooded_road_f1(_fc(ROAD, [True, True]), _fc([]), GSD)
    assert only_pred["precision"] == 0.0 and only_pred["recall"] is None and only_pred["f1"] == 0.0
    only_gt = road_length_f1(_fc([]), _fc(ROAD), GSD)
    assert only_gt["precision"] is None and only_gt["recall"] == 0.0 and only_gt["f1"] == 0.0


def test_aggregate_sums_lengths():
    r1 = road_length_f1(_fc(ROAD), _fc(ROAD), GSD)
    r2 = road_length_f1(_fc([]), _fc(ROAD), GSD)
    agg = aggregate_road_lengths([r1, r2])
    assert agg["n_tiles"] == 2
    assert agg["gt_m"] == pytest.approx(200.0)
    assert agg["recall"] == pytest.approx(0.5)
    assert agg["precision"] == pytest.approx(1.0)
    assert agg["f1"] == pytest.approx(2 / 3)
    assert aggregate_road_lengths([])["f1"] is None


def test_invalid_gsd_raises():
    with pytest.raises(ValueError):
        road_length_f1(_fc(ROAD), _fc(ROAD), 0.0)


def test_dense_noisy_prediction_does_not_blow_up():
    # An untrained model's road "mesh": buffering all of it used to exhaust memory
    import time
    step = 10
    mesh = [[(x, y), (x + step, y)] for x in range(0, 600, step) for y in range(0, 600, step)] + \
           [[(x, y), (x, y + step)] for x in range(0, 600, step) for y in range(0, 600, step)]
    pred = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [list(map(float, p)) for p in l]},
         "properties": {"flooded": False}} for l in mesh]}
    gt = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[0.0, 305.0], [600.0, 305.0]]},
         "properties": {"flooded": False}}]}
    t0 = time.time()
    r = road_length_f1(pred, gt, 0.5)
    assert r["recall"] == pytest.approx(1.0) and r["precision"] < 0.05
    assert time.time() - t0 < 60
