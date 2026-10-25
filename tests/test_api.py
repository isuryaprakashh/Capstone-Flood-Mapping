"""
Integration tests for FastAPI endpoints (/health, /predict).

Dataset-pair endpoints are covered in test_pairs.py.
"""

import io

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_bounds


def _geotiff_bytes(size: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    data = (rng.random((3, size, size)) * 200 + 20).astype(np.uint8)
    buf = io.BytesIO()
    with rasterio.open(
        buf, "w", driver="GTiff", width=size, height=size, count=3, dtype="uint8",
        crs="EPSG:4326", transform=from_bounds(7.039, 50.526, 7.044, 50.531, size, size),
    ) as dst:
        dst.write(data)
    return buf.getvalue()


def test_health_endpoint(api_client: TestClient):
    """Test health check returns status ok and the loaded model."""
    res = api_client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "service" in data
    assert "model" in data and "stub" in data["model"]


def test_predict_upload_returns_vectors_and_summary(api_client: TestClient):
    """Uploading a georeferenced pair returns aligned previews and object-level results."""
    api_client.app.state.predictor = None
    files = {
        "pre_image": ("pre.tif", _geotiff_bytes(96, 0), "image/tiff"),
        "post_image": ("post.tif", _geotiff_bytes(60, 1), "image/tiff"),
    }
    res = api_client.post("/api/predict", files=files)
    assert res.status_code == 200
    data = res.json()

    assert data["ok"] is True
    assert data["mask_shape"] == [5, 96, 96]  # post resampled onto the 96 px pre grid
    assert data["pre_image_base64"] and data["post_image_base64"]
    for key in ("buildings", "flooded_buildings", "road_km", "flooded_road_km", "gsd_m"):
        assert key in data["summary"]
    assert data["building_footprints"]["type"] == "FeatureCollection"
    assert data["road_network"]["type"] == "FeatureCollection"


def test_removed_routing_endpoints_are_gone(api_client: TestClient):
    """The route planner and demo sample endpoints were removed."""
    assert api_client.post("/api/route", data={}).status_code in (404, 405)
    assert api_client.get("/api/sample").status_code in (404, 405)
