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


def _research_run(root, run_id, model="snunet", with_weights=True):
    import json

    import torch
    import yaml

    from models import build_research_model
    from training.engine import model_state

    run = root / run_id
    run.mkdir(parents=True)
    run.joinpath("config.yaml").write_text(yaml.safe_dump({
        "model": model, "model_kwargs": {}, "protocol": "P3", "seed": 0, "amp": False,
        "eval": {"window": 64, "overlap": 16, "window_batch": 2},
    }))
    run.joinpath("metrics_val.json").write_text(json.dumps({"thresholds": {"building": 0.5}}))
    if with_weights:
        torch.save({"model": model_state(build_research_model(model))}, run / "best.pt")


def test_models_list_and_research_model_prediction(api_client: TestClient, tmp_path, monkeypatch):
    """Research runs are listed (available once best.pt exists) and selectable per request."""
    import inference.research as research

    monkeypatch.setattr(research, "RESEARCH_DIR", tmp_path)
    _research_run(tmp_path, "p3_snunet_s0")
    _research_run(tmp_path, "p3_ra_fusion_s0", model="ra_fusion", with_weights=False)

    listed = {m["id"]: m for m in api_client.get("/api/models").json()["models"]}
    assert listed["p3_snunet_s0"]["available"] and not listed["p3_ra_fusion_s0"]["available"]

    files = lambda: {"pre_image": ("pre.tif", _geotiff_bytes(96, 0), "image/tiff"),
                     "post_image": ("post.tif", _geotiff_bytes(96, 1), "image/tiff")}
    res = api_client.post("/api/predict", files=files(), data={"model": "p3_snunet_s0"})
    assert res.status_code == 200
    assert res.json()["model"]["id"] == "p3_snunet_s0" and not res.json()["model"]["stub"]

    missing = api_client.post("/api/predict", files=files(), data={"model": "p3_ra_fusion_s0"})
    assert missing.status_code == 404 and "best.pt" in missing.json()["detail"]
    assert api_client.post("/api/predict", files=files(), data={"model": "../x"}).status_code == 404
