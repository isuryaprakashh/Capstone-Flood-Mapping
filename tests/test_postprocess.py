"""
Tests for prediction vectorization (buildings, road graph, summary).
"""

import numpy as np
import pytest

from inference.postprocess import (
    build_road_graph,
    extract_buildings,
    road_graph_to_geojson,
    summarize,
    vectorize_prediction,
)


def test_extract_buildings_attributes_flood_by_majority():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:30] = 1          # building A
    mask[50:70, 50:80] = 1          # building B
    mask[90, 90] = 1                # speckle, removed
    flooded = np.zeros_like(mask)
    flooded[10:30, 10:25] = 1       # 75% of A flooded
    flooded[50:55, 50:80] = 1       # 25% of B flooded

    fc = extract_buildings(mask, flooded, gsd_m=0.5)
    feats = sorted(fc["features"], key=lambda f: f["properties"]["area_m2"])
    assert len(feats) == 2
    assert feats[0]["properties"]["flooded"] is True    # A (400 px)
    assert feats[1]["properties"]["flooded"] is False   # B (600 px)
    assert feats[0]["properties"]["area_m2"] == pytest.approx(400 * 0.25, rel=0.1)


def test_extract_buildings_empty_mask():
    fc = extract_buildings(np.zeros((32, 32), dtype=np.uint8), None)
    assert fc["features"] == []


def _plus_road(size=101, half_width=2):
    road = np.zeros((size, size), dtype=np.uint8)
    c = size // 2
    road[c - half_width:c + half_width + 1, 5:size - 5] = 1
    road[5:size - 5, c - half_width:c + half_width + 1] = 1
    return road, c


def test_road_graph_splits_at_junction_and_marks_flooded_arm():
    road, c = _plus_road()
    flooded = np.zeros_like(road)
    flooded[:c - 5, :] = road[:c - 5, :]  # flood the northern arm only

    G = build_road_graph(road, flooded, gsd_m=0.5, flood_dilation_px=1, min_component_m=1.0)
    lengths = [d["length_m"] for _, _, d in G.edges(data=True)]
    flooded_edges = [d for _, _, d in G.edges(data=True) if d["flooded"]]

    # Junction pixel clusters merge into one node: exactly four arms
    assert G.number_of_edges() == 4
    assert sorted(dict(G.degree()).values()) == [1, 1, 1, 1, 4]
    assert len(flooded_edges) == 1
    assert sum(lengths) == pytest.approx(2 * 90 * 0.5, rel=0.15)

    fc = road_graph_to_geojson(G)
    assert len(fc["features"]) == G.number_of_edges()
    assert all(len(f["geometry"]["coordinates"]) >= 2 for f in fc["features"])


def test_road_graph_drops_short_fragments():
    road = np.zeros((60, 60), dtype=np.uint8)
    road[30, 10:14] = 1                    # 2 m fragment at 0.5 m/px
    G = build_road_graph(road, None, gsd_m=0.5, min_component_m=10.0)
    assert G.number_of_edges() == 0


def test_empty_prediction_has_no_invented_roads():
    binary = np.zeros((5, 64, 64), dtype=np.uint8)
    out = vectorize_prediction({"binary_mask": binary}, gsd_m=0.5)
    assert out["roads"]["features"] == []
    assert out["buildings"]["features"] == []
    assert out["summary"]["road_km"] == 0.0


def test_summarize_counts_and_lengths():
    buildings = {"features": [
        {"properties": {"flooded": True}}, {"properties": {"flooded": False}},
    ]}
    roads = {"features": [
        {"properties": {"flooded": True, "length_m": 250.0}},
        {"properties": {"flooded": False, "length_m": 750.0}},
    ]}
    s = summarize(buildings, roads)
    assert s["buildings"] == 2 and s["flooded_buildings"] == 1
    assert s["flooded_building_pct"] == 50.0
    assert s["road_km"] == 1.0 and s["flooded_road_km"] == 0.25
    assert s["flooded_road_pct"] == 25.0
