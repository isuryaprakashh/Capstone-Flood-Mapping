"""
Tests for pixel metrics, count arithmetic, bootstrap intervals and stress helpers.
"""

import numpy as np
import pytest

from evaluation.bootstrap import bootstrap_ci
from evaluation.common import prf_from_counts, sum_counts
from evaluation.pixels import class_pixel_counts, pixel_counts, pixel_metrics
from evaluation.stress import shift_image, shift_schedule


# ─── Pixels and counts ──────────────────────────────────────────────────────


def test_pixel_counts_with_valid_mask():
    pred = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    target = np.array([[1, 0, 1, 0]], dtype=np.uint8)
    assert pixel_counts(pred, target) == {"tp": 1, "fp": 1, "fn": 1}
    valid = np.array([[True, False, True, True]])
    assert pixel_counts(pred, target, valid) == {"tp": 1, "fp": 0, "fn": 1}
    with pytest.raises(ValueError):
        pixel_counts(pred, target[:, :3])


def test_pixel_metrics_none_for_absent_class():
    pred = np.zeros((3, 8, 8), dtype=np.uint8)
    target = np.zeros((3, 8, 8), dtype=np.uint8)
    pred[0, :4] = 1
    target[0, :2] = 1
    target[2, 0, 0] = 1
    counts = class_pixel_counts(pred, target, ["a", "b", "c"])
    m = pixel_metrics(counts)
    assert m["a"]["iou"] == pytest.approx(0.5)
    assert m["a"]["precision"] == pytest.approx(0.5)
    assert m["a"]["recall"] == pytest.approx(1.0)
    assert m["a"]["f1"] == pytest.approx(2 / 3)
    assert all(m["b"][k] is None for k in ("iou", "f1", "precision", "recall"))
    assert m["c"]["iou"] == 0.0 and m["c"]["recall"] == 0.0 and m["c"]["precision"] is None


def test_sum_counts_nested_and_micro_average():
    a = {"x": {"tp": 1, "fp": 2, "fn": 0}, "n": 1}
    b = {"x": {"tp": 3, "fp": 0, "fn": 4}, "n": 2, "extra": 5}
    total = sum_counts([a, b])
    assert total == {"x": {"tp": 4, "fp": 2, "fn": 4}, "n": 3, "extra": 5}
    assert prf_from_counts(**total["x"])["f1"] == pytest.approx(8 / 14)
    with pytest.raises(TypeError):
        sum_counts([{"flag": True}])


# ─── Bootstrap ──────────────────────────────────────────────────────────────


def test_bootstrap_ci_contains_estimate_and_is_deterministic():
    values = list(np.random.default_rng(1).normal(0.6, 0.1, size=60))
    a = bootstrap_ci(values, n=500, seed=7)
    b = bootstrap_ci(values, n=500, seed=7)
    c = bootstrap_ci(values, n=500, seed=8)
    assert a == b
    assert a != c
    assert a["low"] <= a["estimate"] <= a["high"]
    assert a["estimate"] == pytest.approx(np.mean(values))
    assert a["n_boot"] == 500 and a["n_tiles"] == 60
    assert a["high"] - a["low"] == pytest.approx(2 * 1.96 * np.std(values, ddof=1) / np.sqrt(60), rel=0.25)


def test_bootstrap_ratio_metric_resamples_counts():
    rng = np.random.default_rng(0)
    tiles = [{"tp": int(t), "fp": int(f), "fn": int(n)} for t, f, n in rng.integers(0, 20, size=(40, 3))]

    def micro_f1(sample):
        return prf_from_counts(**sum_counts(sample))["f1"]

    result = bootstrap_ci(tiles, micro_f1, n=300, seed=0)
    assert result["estimate"] == pytest.approx(micro_f1(tiles))
    assert result["low"] <= result["estimate"] <= result["high"]
    assert result["high"] - result["low"] > 0


def test_bootstrap_handles_none_and_empty():
    assert bootstrap_ci([0.5, None, 0.7], n=50)["estimate"] == pytest.approx(0.6)
    empty = bootstrap_ci([], n=50)
    assert empty["estimate"] is None and empty["low"] is None and empty["n_boot"] == 0
    undefined = bootstrap_ci([None, None], n=20)
    assert undefined["estimate"] is None and undefined["low"] is None
    with pytest.raises(ValueError):
        bootstrap_ci([1.0], alpha=1.5)


# ─── Stress helpers ─────────────────────────────────────────────────────────


def test_shift_image_moves_content_with_zero_fill():
    img = np.arange(1, 2 * 4 * 5 + 1, dtype=np.float32).reshape(2, 4, 5)
    out = shift_image(img, 1, 2)
    assert out.shape == img.shape and out.dtype == img.dtype
    np.testing.assert_array_equal(out[:, 1:, 2:], img[:, :-1, :-2])
    assert np.all(out[:, 0, :] == 0) and np.all(out[:, :, :2] == 0)
    back = shift_image(img, -1, -2)
    np.testing.assert_array_equal(back[:, :-1, :-2], img[:, 1:, 2:])
    np.testing.assert_array_equal(shift_image(img, 0, 0), img)


def test_shift_image_edge_cases():
    mask = np.ones((6, 6), dtype=bool)
    valid = shift_image(mask, 0, -3)
    assert valid.sum() == 18 and not valid[:, 3:].any()
    assert not shift_image(mask, 6, 0).any()
    assert shift_image(mask, 2.0, 0).sum() == 24
    with pytest.raises(ValueError):
        shift_image(mask, 1.5, 0)


def test_shift_schedule():
    schedule = shift_schedule(max_px=40, step=5)
    assert schedule[0] == (0, 0, 0)
    assert len(schedule) == 1 + 8 * 8
    assert sorted({m for m, _, _ in schedule}) == list(range(0, 41, 5))
    for m, dy, dx in schedule:
        assert np.hypot(dy, dx) == pytest.approx(m, abs=0.75)
    assert shift_schedule(10, 10, directions=4) == [(0, 0, 0), (10, 0, 10), (10, -10, 0), (10, 0, -10), (10, 10, 0)]
    with pytest.raises(ValueError):
        shift_schedule(step=0)
