"""
Tests for the pre/post pair index and the dataset-pair API endpoints.
"""

import csv
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from dataloader.pairs import PairIndex

BOUNDS = (7.0390, 50.5260, 7.0440, 50.5310)


def _write_tif(path, size, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    data = (rng.random((3, size, size)) * 200 + 20).astype(np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", width=size, height=size, count=3, dtype="uint8",
        crs="EPSG:4326", transform=from_bounds(*BOUNDS, size, size),
    ) as dst:
        dst.write(data)


@pytest.fixture
def dataset_root(tmp_path):
    """Two tiny AOIs: one labelled with a two-post tile, one unlabelled."""
    aoi = tmp_path / "Germany_Training_Public"
    (aoi / "annotations").mkdir(parents=True)
    rows = [
        ["0_1_1.geojson", "PRE_A_0_1_1.tif", "POSTX_0_1_1.tif", "POSTY_0_1_1.tif"],
        ["0_1_2.geojson", "PRE_A_0_1_2.tif", "POSTX_0_1_2.tif", ""],
    ]
    with open(aoi / "Germany_Training_Public_label_image_mapping.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "pre-event image", "post-event image 1", "post-event image 2"])
        w.writerows(rows)
    for r in rows:
        _write_tif(aoi / "PRE-event" / r[1], 64)
        for post in r[2:]:
            if post:
                _write_tif(aoi / "POST-event" / post, 40, seed=1)

    # Ground truth for tile 0_1_2: one flooded building, one flooded road
    (aoi / "annotations" / "0_1_2.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"building": "yes", "flooded": "yes"},
            "geometry": {"type": "Polygon", "coordinates": [[
                [7.0400, 50.5270], [7.0410, 50.5270], [7.0410, 50.5280],
                [7.0400, 50.5280], [7.0400, 50.5270],
            ]]},
        }],
    }))
    with open(aoi / "Germany_Training_Public_reference.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ImageId", "Object", "Wkt_Pix", "Flooded", "length_m", "travel_time_s"])
        w.writerow(["PRE_A_0_1_2", "Road", "LINESTRING (5 5, 60 60)", "True", "30", "3"])

    test_aoi = tmp_path / "Louisiana-West_Test_Public"
    test_aoi.mkdir(parents=True)
    with open(test_aoi / "Louisiana-West_Test_Public_label_image_mapping.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "pre-event image", "post-event image 1", "post-event image 2"])
        w.writerow(["0_9_9.geojson", "PRE_W_0_9_9.tif", "POSTW_0_9_9.tif", ""])
    _write_tif(test_aoi / "PRE-event" / "PRE_W_0_9_9.tif", 64)
    _write_tif(test_aoi / "POST-event" / "POSTW_0_9_9.tif", 40)

    stats = tmp_path / "alignment_stats.csv"
    with open(stats, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["aoi", "tile", "post_index", "shift_px"])
        w.writerow(["Germany_Training_Public", "0_1_1", 1, 23.0])
        w.writerow(["Germany_Training_Public", "0_1_1", 2, 4.3])
    return tmp_path


def test_index_lists_aois_and_pairs(dataset_root):
    index = PairIndex(dataset_root)
    aois = {a["aoi"]: a for a in index.aois()}
    assert aois["Germany_Training_Public"]["num_pairs"] == 2
    assert aois["Germany_Training_Public"]["has_labels"] is True
    assert aois["Louisiana-West_Test_Public"]["has_labels"] is False


def test_lookup_returns_mapped_post(dataset_root):
    index = PairIndex(dataset_root)
    pair = index.lookup("C:/some/folder/PRE_A_0_1_2.tif")  # path components ignored
    assert pair["aoi"] == "Germany_Training_Public"
    assert pair["post_images"] == ["POSTX_0_1_2.tif"]
    assert index.lookup("unknown.tif") is None


def test_posts_ordered_best_aligned_first(dataset_root):
    index = PairIndex(dataset_root, alignment_stats=dataset_root / "alignment_stats.csv")
    pair = index.lookup("PRE_A_0_1_1.tif")
    assert pair["post_images"] == ["POSTY_0_1_1.tif", "POSTX_0_1_1.tif"]
    assert pair["post_shift_px"] == [4.3, 23.0]


def test_search_filters_by_aoi_and_query(dataset_root):
    index = PairIndex(dataset_root)
    pairs, total = index.search(aoi="Germany_Training_Public", query="0_1_2")
    assert total == 1 and pairs[0]["pre_image"] == "PRE_A_0_1_2.tif"


def test_resolve_only_allows_indexed_files(dataset_root):
    index = PairIndex(dataset_root)
    assert index.resolve("Germany_Training_Public", "post", "POSTX_0_1_2.tif").exists()
    assert index.resolve("Germany_Training_Public", "post", "../../alignment_stats.csv") is None
    # Directory components are stripped: the resolved path is always the indexed file
    expected = dataset_root / "Germany_Training_Public" / "PRE-event" / "PRE_A_0_1_2.tif"
    assert index.resolve("Germany_Training_Public", "pre", "../../PRE_A_0_1_2.tif") == expected
    # A pre image can only be resolved inside its own AOI
    assert index.resolve("Louisiana-West_Test_Public", "pre", "PRE_A_0_1_2.tif") is None


@pytest.fixture
def pair_client(dataset_root, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.main import app

    monkeypatch.setenv("SPACENET8_DATA_DIR", str(dataset_root))
    app.state.pair_index = None
    app.state.predictor = None
    yield TestClient(app)
    app.state.pair_index = None


def test_api_lookup_and_404(pair_client):
    res = pair_client.get("/api/pairs/lookup", params={"pre_image": "PRE_A_0_1_2.tif"})
    assert res.status_code == 200
    assert res.json()["pair"]["post_images"] == ["POSTX_0_1_2.tif"]

    assert pair_client.get("/api/pairs/lookup", params={"pre_image": "nope.tif"}).status_code == 404


def test_api_image_download_and_traversal(pair_client):
    res = pair_client.get(
        "/api/pairs/image",
        params={"aoi": "Germany_Training_Public", "kind": "post", "name": "POSTX_0_1_2.tif"},
    )
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/tiff"

    bad = pair_client.get(
        "/api/pairs/image",
        params={"aoi": "Germany_Training_Public", "kind": "post", "name": "../../../.env"},
    )
    assert bad.status_code == 404


def test_api_thumbnail(pair_client):
    res = pair_client.get(
        "/api/pairs/thumbnail",
        params={"aoi": "Germany_Training_Public", "kind": "pre", "name": "PRE_A_0_1_1.tif"},
    )
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"


def test_api_predict_pair_autoselects_post(pair_client):
    res = pair_client.post(
        "/api/predict_pair",
        data={"aoi": "Germany_Training_Public", "pre_image": "PRE_A_0_1_2.tif"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["pair"]["post_image"] == "POSTX_0_1_2.tif"
    assert data["pre_image_base64"] and data["post_image_base64"]

    gt = data["ground_truth"]
    assert gt["summary"]["buildings"] == 1 and gt["summary"]["flooded_buildings"] == 1
    assert len(gt["road_network"]["features"]) == 1
    assert gt["road_network"]["features"][0]["properties"]["flooded"] is True
    assert {"iou_building", "f1_flooded_road"} <= set(gt["metrics"])


def test_api_predict_pair_unlabelled_has_no_ground_truth(pair_client):
    res = pair_client.post(
        "/api/predict_pair",
        data={"aoi": "Louisiana-West_Test_Public", "pre_image": "PRE_W_0_9_9.tif"},
    )
    assert res.status_code == 200
    assert res.json()["ground_truth"] is None


def test_api_predict_pair_rejects_wrong_aoi(pair_client):
    res = pair_client.post(
        "/api/predict_pair",
        data={"aoi": "Louisiana-West_Test_Public", "pre_image": "PRE_A_0_1_2.tif"},
    )
    assert res.status_code == 404
