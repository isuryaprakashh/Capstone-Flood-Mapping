"""
Tests for SpaceNet 8 dataset preparation (post choice, targets, splits).
"""

import csv
import json

import cv2
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from dataloader.align import apply_shift
from dataloader.prepare import (
    TARGET_CHANNELS,
    assign_folds,
    load_tile,
    make_splits,
    prepare_dataset,
    road_junctions,
)

SIZE = 650


def _texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = (rng.random((3, SIZE, SIZE)) * 220 + 20).astype(np.uint8)
    return np.stack([cv2.GaussianBlur(b, (0, 0), 2) for b in img])


def _write_tif(path, data, bounds):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", width=SIZE, height=SIZE, count=3, dtype="uint8",
        crs="EPSG:4326", transform=from_bounds(*bounds, SIZE, SIZE), nodata=255,
    ) as dst:
        dst.write(data)


def _line(x0, y0, x1, y1, flooded=False):
    return {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[x0, y0], [x1, y1]]},
            "properties": {"flooded": flooded}}


def _make_aoi(root, aoi, tiles, lon0, lat0):
    """Write a tiny AOI: each tile gets a building, a T-junction road and posts."""
    d = root / aoi
    (d / "annotations").mkdir(parents=True)
    rows, ref = [], []
    for i, tile in enumerate(tiles):
        west, south = lon0 + i * 0.05, lat0
        bounds = (west, south, west + 0.002, south + 0.002)
        pre = _texture(i)
        pre_name, post_a, post_b = f"PRE_{tile}.tif", f"POSTA_{tile}.tif", f"POSTB_{tile}.tif"
        _write_tif(d / "PRE-event" / pre_name, pre, bounds)
        _write_tif(d / "POST-event" / post_a, apply_shift(pre, 9.0, -7.0), bounds)  # misaligned
        _write_tif(d / "POST-event" / post_b, pre, bounds)                         # aligned
        rows.append([f"{tile}.geojson", pre_name, post_a, post_b])

        # Building covering pixels ~[100, 200] in x and y (geographic coordinates)
        px = (bounds[2] - bounds[0]) / SIZE
        x0, x1 = west + 100 * px, west + 200 * px
        y0, y1 = bounds[3] - 100 * px, bounds[3] - 200 * px
        (d / "annotations" / f"{tile}.geojson").write_text(json.dumps({
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"building": "yes", "flooded": "yes" if i == 0 else None},
                "geometry": {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]},
            }],
        }))
        # T junction at (400, 300): a through road and a branch ending on it
        ref.append([pre_name.replace(".tif", ""), "Road", "LINESTRING (300 300, 500 300)", "True" if i == 0 else "False", "60", "5"])
        ref.append([pre_name.replace(".tif", ""), "Road", "LINESTRING (400 300, 400 500)", "False", "60", "5"])

    with open(d / f"{aoi}_label_image_mapping.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "pre-event image", "post-event image 1", "post-event image 2"])
        w.writerows(rows)
    with open(d / f"{aoi}_reference.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ImageId", "Object", "Wkt_Pix", "Flooded", "length_m", "travel_time_s"])
        w.writerows(ref)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    _make_aoi(root, "Germany_Training_Public", ["0_1_1", "0_1_2"], 7.00, 50.50)
    _make_aoi(root, "Louisiana-East_Training_Public", ["1_1_1", "1_1_2", "1_1_3"], -90.10, 29.70)
    out = tmp_path_factory.mktemp("processed")
    manifest = prepare_dataset(root, out, workers=1)
    return out, manifest, root


def test_all_tiles_prepared(prepared):
    out, manifest, _ = prepared
    assert manifest["n_tiles"] == 5 and manifest["n_failed"] == 0
    for meta in manifest["tiles"]:
        tile_dir = out / "tiles" / meta["id"]
        for name in ("pre.png", "post.png", "targets.npz", "gt.json", "meta.json"):
            assert (tile_dir / name).exists()


def test_best_aligned_post_is_chosen(prepared):
    _, manifest, _ = prepared
    for meta in manifest["tiles"]:
        assert meta["post_image"].startswith("POSTB_")  # the aligned one
        shifts = {c["name"][:5]: c["shift_px"] for c in meta["post_candidates"]}
        assert shifts["POSTA"] == pytest.approx(np.hypot(9, 7), abs=0.5)
        assert shifts["POSTB"] < 0.5


def test_targets_are_consistent(prepared):
    out, manifest, _ = prepared
    tile = load_tile(out, manifest["tiles"][0]["id"])
    t = tile["targets"]
    assert set(t) == set(TARGET_CHANNELS)
    assert tile["pre"].shape == tile["post"].shape == (3, SIZE, SIZE)
    assert 0.015 < t["building"].mean() < 0.04             # ~100x100 px building
    assert t["building_border"].sum() > 0 and not (t["building_border"] & ~t["building"]).any()
    assert not (t["flooded_building"] & ~t["building"]).any()
    assert not (t["flooded_road"] & ~t["road"]).any()
    assert t["junction"][300, 400] and t["junction"].sum() > 0
    assert t["valid"].all()


def test_flood_attributes_follow_labels(prepared):
    out, manifest, _ = prepared
    metas = {m["id"]: m for m in manifest["tiles"]}
    flooded = load_tile(out, "Germany_Training_Public/0_1_1")["targets"]
    dry = load_tile(out, "Germany_Training_Public/0_1_2")["targets"]
    assert flooded["flooded_building"].any() and flooded["flooded_road"].any()
    assert not dry["flooded_building"].any() and not dry["flooded_road"].any()
    assert metas["Germany_Training_Public/0_1_1"]["has_flood"] is True
    assert metas["Germany_Training_Public/0_1_2"]["has_flood"] is False


def test_splits_are_disjoint_and_complete(prepared):
    out, manifest, _ = prepared
    splits = json.loads((out / "splits.json").read_text())
    ger = {m["id"] for m in manifest["tiles"] if m["aoi"].startswith("Germany")}
    lae = {m["id"] for m in manifest["tiles"] if m["aoi"].startswith("Louisiana-East")}

    assert set(splits["P1"]["test"]) == ger
    assert set(splits["P1"]["train"]) | set(splits["P1"]["val"]) == lae
    assert set(splits["P2"]["test"]) == lae
    for name in ("P1", "P2", "P3"):
        s = splits[name]
        assert not (set(s["train"]) & set(s["val"])) and not (set(s["train"]) & set(s["test"]))
        assert not (set(s["val"]) & set(s["test"]))


def test_prepare_resumes_without_reprocessing(prepared, monkeypatch):
    out, manifest, root = prepared
    import dataloader.prepare as prep

    def fail(_):
        raise AssertionError("tile was reprocessed")

    monkeypatch.setattr(prep, "prepare_tile", fail)
    again = prep.prepare_dataset(root, out, workers=1)  # every meta.json is reused
    assert again["n_tiles"] == manifest["n_tiles"]


def test_road_junctions_t_cross_and_corner():
    t_junction = {"features": [_line(0, 50, 100, 50), _line(50, 50, 50, 100)]}
    cross = {"features": [_line(0, 50, 50, 50), _line(50, 50, 100, 50), _line(50, 0, 50, 50), _line(50, 50, 50, 100)]}
    corner = {"features": [_line(0, 0, 50, 0), _line(50, 0, 50, 50)]}
    assert road_junctions(t_junction) == [(50.0, 50.0)]
    assert road_junctions(cross) == [(50.0, 50.0)]
    assert road_junctions(corner) == []


def test_folds_keep_neighbours_together_and_are_deterministic():
    metas = []
    for i in range(40):
        lon = -90.0 + (i % 8) * 0.005   # 8 columns, 0.005 deg apart: 4 per 0.02-deg block
        lat = 29.7 + (i // 8) * 0.005
        metas.append({"id": f"A/{i}", "aoi": "A", "centroid_lonlat": [lon, lat], "has_flood": i % 3 == 0,
                      "labelled": True})
    folds = assign_folds(metas, n_folds=5, block_deg=0.02, seed=1)
    assert folds == assign_folds(metas, n_folds=5, block_deg=0.02, seed=1)

    block = lambda m: (int(np.floor(m["centroid_lonlat"][1] / 0.02)), int(np.floor(m["centroid_lonlat"][0] / 0.02)))
    by_block = {}
    for m in metas:
        by_block.setdefault(block(m), set()).add(folds[m["id"]])
    assert all(len(f) == 1 for f in by_block.values())  # a block never straddles folds
    assert len(set(folds.values())) > 1
