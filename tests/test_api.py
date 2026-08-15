"""
Integration tests for FastAPI endpoints (/health, /sample, /predict, /route).
"""

import io
import pytest
from fastapi.testclient import TestClient


def test_health_endpoint(api_client: TestClient):
    """Test health check returns status ok."""
    res = api_client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "service" in data


def test_sample_endpoint(api_client: TestClient):
    """Test sample prediction returns valid demo statistics and base64 mask."""
    res = api_client.get("/api/sample")
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert "statistics" in data
    assert "mask_png_base64" in data
    assert len(data["class_names"]) == 5


def test_route_endpoint(api_client: TestClient):
    """Test routing endpoint generates a path and distance."""
    payload = {
        "start_y": 100,
        "start_x": 100,
        "end_y": 400,
        "end_x": 400,
    }
    res = api_client.post("/api/route", data=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert "route" in data
    assert "distance_m" in data
    assert "route_type" in data
