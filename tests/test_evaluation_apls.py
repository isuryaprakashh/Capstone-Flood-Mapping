"""
Tests for APLS road-network scoring (evaluation.apls).
"""

import itertools
import time

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, Point
from shapely.ops import substring

from evaluation.apls import (
    _connected_pairs,
    _control_graph,
    _network,
    _triu_pair,
    apls,
    apls_detailed,
    apls_flood,
    build_gt_graph,
    flood_subgraph,
    prediction_graph_from_postprocess,
)
from inference.postprocess import build_road_graph

GSD = 0.5


def _fc(lines, flooded=None):
    flooded = flooded or [False] * len(lines)
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [list(map(float, p)) for p in line]},
                "properties": {"flooded": bool(f)},
            }
            for line, f in zip(lines, flooded)
        ],
    }


def _straight_segments(n=10, seg_px=40, x0=10, y=100):
    """A 200 m road (at 0.5 m/px) cut into n reference-style segments."""
    return [[(x0 + seg_px * i, y), (x0 + seg_px * (i + 1), y)] for i in range(n)]


def _grid_segments(n_streets=4, spacing_px=300, seg_px=100, origin=100):
    """Square street grid cut into short segments that meet at shared endpoints."""
    extent = spacing_px * (n_streets - 1)
    segs = []
    for k in range(n_streets):
        c = origin + k * spacing_px
        for s in range(0, extent, seg_px):
            segs.append([(origin + s, c), (origin + s + seg_px, c)])
            segs.append([(c, origin + s), (c, origin + s + seg_px)])
    return segs


def _shift(segs, dx, dy):
    return [[(x + dx, y + dy) for x, y in seg] for seg in segs]


# ─── Graph construction ─────────────────────────────────────────────────────


def test_build_gt_graph_snaps_close_endpoints_and_measures_length():
    G = build_gt_graph(_fc([[(0, 0), (100, 0)], [(101, 1), (200, 1)]]), gsd_m=GSD, snap_px=2.0)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    assert nx.is_connected(G)
    total_m = sum(d["length_m"] for _, _, d in G.edges(data=True))
    assert total_m == pytest.approx(200 * GSD, abs=0.5)
    for _, _, d in G.edges(data=True):
        assert d["length_m"] == pytest.approx(d["length_px"] * GSD)


def test_build_gt_graph_keeps_endpoints_beyond_snap_apart():
    G = build_gt_graph(_fc([[(0, 0), (100, 0)], [(105, 0), (200, 0)]]), gsd_m=GSD, snap_px=2.0)
    assert G.number_of_nodes() == 4
    assert nx.number_connected_components(G) == 2


@pytest.mark.parametrize("stem_end_y", [50.0, 51.5])
def test_build_gt_graph_splits_segment_at_t_junction(stem_end_y):
    G = build_gt_graph(_fc([[(0, 50), (100, 50)], [(50, stem_end_y), (50, 0)]]), gsd_m=GSD, snap_px=2.0)
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 3
    assert sorted(dict(G.degree()).values()) == [1, 1, 1, 3]
    junction = next(n for n in G.nodes if G.degree(n) == 3)
    assert G.nodes[junction]["x"] == pytest.approx(50.0)
    assert G.nodes[junction]["y"] == pytest.approx(stem_end_y)


def test_build_gt_graph_drops_duplicates_and_keeps_crossings_unconnected():
    seg = [(0, 50), (100, 50)]
    G = build_gt_graph(_fc([seg, seg[::-1], [(50, 0), (50, 100)]]), gsd_m=GSD)
    assert G.number_of_edges() == 2
    assert G.number_of_nodes() == 4
    assert nx.number_connected_components(G) == 2


def test_build_gt_graph_keeps_flood_flags_and_empty_input():
    G = build_gt_graph(_fc([[(0, 0), (50, 0)], [(50, 0), (100, 0)]], [True, False]), gsd_m=GSD)
    assert sorted(d["flooded"] for _, _, d in G.edges(data=True)) == [False, True]
    assert build_gt_graph(_fc([]), gsd_m=GSD).number_of_edges() == 0


def test_prediction_graph_from_postprocess_converts_coordinates():
    P = nx.MultiGraph()
    P.add_edge((10, 0), (10, 30), path=[(10, 30), (10, 15), (10, 0)], length_m=15.0, flooded=True)
    H = prediction_graph_from_postprocess(P, gsd_m=GSD)
    assert H.number_of_nodes() == 2 and H.number_of_edges() == 1
    (u, v, d), = H.edges(data=True)
    assert d["length_px"] == pytest.approx(30.0)
    assert d["length_m"] == pytest.approx(15.0)
    assert d["flooded"] is True
    xs = sorted(H.nodes[n]["x"] for n in H.nodes)
    assert xs == [0.5, 30.5]
    assert H.nodes[u]["y"] == pytest.approx(10.5)
    start = d["geometry"].coords[0]
    assert start == pytest.approx((H.nodes[u]["x"], H.nodes[u]["y"]))


# ─── APLS ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("segs", [_straight_segments(), _grid_segments()])
def test_identical_graphs_score_one(segs):
    G = build_gt_graph(_fc(segs), gsd_m=GSD)
    assert apls(G, G, GSD) == pytest.approx(1.0)
    assert apls(G, G, GSD, max_pairs=None) == pytest.approx(1.0)


def test_segmentation_granularity_does_not_change_score():
    fine = build_gt_graph(_fc(_straight_segments(n=10)), gsd_m=GSD)
    coarse = build_gt_graph(_fc([[(10, 100), (410, 100)]]), gsd_m=GSD)
    result = apls_detailed(fine, coarse, GSD)
    assert result["apls"] == pytest.approx(1.0)
    assert result["n_control_gt"] == result["n_control_prop"] == 5


def test_removed_middle_segment_breaks_paths():
    segs = _straight_segments()
    G = build_gt_graph(_fc(segs), gsd_m=GSD)
    P = build_gt_graph(_fc(segs[:5] + segs[6:]), gsd_m=GSD)
    result = apls_detailed(G, P, GSD)
    # Control points at 0/50/100/150/200 m; only 4 of 10 GT pairs avoid the gap
    assert result["apls_gt_to_prop"] == pytest.approx(0.4)
    assert result["apls_prop_to_gt"] == pytest.approx(1.0)
    assert result["apls"] == pytest.approx(2 * 0.4 * 1.0 / 1.4)


def test_removed_segments_in_grid_lower_score():
    segs = _grid_segments()
    G = build_gt_graph(_fc(segs), gsd_m=GSD)
    P = build_gt_graph(_fc([s for i, s in enumerate(segs) if i not in (4, 5, 30)]), gsd_m=GSD)
    score = apls(G, P, GSD, max_pairs=None)
    assert 0.3 < score < 0.98


def test_shifted_copy_within_buffer_scores_close_to_one():
    segs = _grid_segments()
    G = build_gt_graph(_fc(segs), gsd_m=GSD)
    P = build_gt_graph(_fc(_shift(segs, 2.0, 2.0)), gsd_m=GSD)
    assert apls(G, P, GSD) > 0.95


def test_parallel_copy_beyond_buffer_scores_zero():
    segs = _straight_segments()
    G = build_gt_graph(_fc(segs), gsd_m=GSD)
    P = build_gt_graph(_fc(_shift(segs, 0.0, 20.0)), gsd_m=GSD)  # 10 m away, buffer 4 m
    result = apls_detailed(G, P, GSD)
    assert result["apls_gt_to_prop"] == 0.0
    assert result["apls"] == 0.0


def test_empty_graphs():
    G = build_gt_graph(_fc(_straight_segments()), gsd_m=GSD)
    empty = nx.MultiGraph()
    assert apls(G, empty, GSD) == 0.0
    assert apls(empty, G, GSD) == 0.0
    assert apls(empty, empty, GSD) == 1.0
    assert apls(_fc([]), _fc([]), GSD) == 1.0


def test_spurious_disconnected_prediction_is_penalized():
    segs = _straight_segments()
    G = build_gt_graph(_fc(segs), gsd_m=GSD)
    P = build_gt_graph(_fc(segs + _shift(segs, 0.0, 600.0)), gsd_m=GSD)
    result = apls_detailed(G, P, GSD, max_pairs=None)
    assert result["apls_gt_to_prop"] == pytest.approx(1.0)
    assert result["apls_prop_to_gt"] < 1.0
    assert result["apls"] < 1.0


def test_apls_accepts_feature_collections_and_postprocess_graphs():
    road = np.zeros((101, 101), dtype=np.uint8)
    road[48:53, 5:96] = 1
    road[5:96, 48:53] = 1
    P = build_road_graph(road, gsd_m=GSD, min_component_m=1.0)
    c = (50.5, 50.5)
    gt = _fc([[c, (5, 50.5)], [c, (96, 50.5)], [c, (50.5, 5)], [c, (50.5, 96)]])
    score = apls(gt, P, GSD)
    assert score > 0.85
    assert score == pytest.approx(apls(build_gt_graph(gt, GSD), prediction_graph_from_postprocess(P, GSD), GSD))


def test_pair_sampling_is_deterministic_and_close_to_exact():
    segs = _grid_segments()
    G = build_gt_graph(_fc(segs), gsd_m=GSD)
    P = build_gt_graph(_fc([s for i, s in enumerate(segs) if i not in (4, 5, 30)]), gsd_m=GSD)
    a = apls_detailed(G, P, GSD, max_pairs=200, seed=3)
    b = apls_detailed(G, P, GSD, max_pairs=200, seed=3)
    exact = apls_detailed(G, P, GSD, max_pairs=None)
    assert a == b
    assert a["n_pairs_gt"] == 200
    assert exact["n_pairs_gt"] > 200
    assert a["apls"] == pytest.approx(exact["apls"], abs=0.06)


def _naive_direction(A, B, gsd, buffer_m=4.0, spacing_m=50.0):
    """Unvectorized APLS(A -> B): sequential point insertion, networkx all-pairs paths."""

    def oriented_edges(G):
        xy = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}
        edges = []
        for u, v, d in G.edges(data=True):
            g = d["geometry"]
            start = Point(g.coords[0])
            if start.distance(Point(xy[u])) > start.distance(Point(xy[v])):
                g = LineString(list(g.coords)[::-1])
            edges.append((u, v, g))
        return xy, edges

    def all_pairs(xy, edges):
        S = nx.Graph()
        S.add_nodes_from(xy)
        for u, v, g in edges:
            w = g.length * gsd
            if u != v and (not S.has_edge(u, v) or S.edges[u, v]["w"] > w):
                S.add_edge(u, v, w=w)
        return dict(nx.all_pairs_dijkstra_path_length(S, weight="w"))

    xy_a, edges_a = oriented_edges(A)
    ctrl_edges, counter = [], itertools.count()
    for u, v, g in edges_a:
        length = g.length * gsd
        if length < 0.75 * spacing_m:
            ds = []
        elif length <= spacing_m:
            ds = [length / 2]
        else:
            ds = list(np.linspace(0, length, len(np.arange(0, length, spacing_m)) + 1)[1:-1])
        pos = [0.0] + [d / gsd for d in ds] + [g.length]
        chain = [u] + [("mid", next(counter)) for _ in ds] + [v]
        for n, p in zip(chain[1:-1], pos[1:-1]):
            xy_a[n] = g.interpolate(p).coords[0]
        ctrl_edges += [(chain[k], chain[k + 1], substring(g, pos[k], pos[k + 1])) for k in range(len(chain) - 1)]

    xy_b, edges_b = oriented_edges(B)
    tol, snapped = 0.05 / gsd, {}
    for n, p in xy_a.items():
        pt = Point(p)
        candidates = [(g.distance(pt), i) for i, (_, _, g) in enumerate(edges_b)]
        if not candidates or min(candidates)[0] > buffer_m / gsd:
            continue
        i = min(candidates)[1]
        u, v, g = edges_b[i]
        d = g.project(pt)
        if d <= tol:
            snapped[n] = u
        elif d >= g.length - tol:
            snapped[n] = v
        else:
            snapped[n] = ("snap", n)
            xy_b[snapped[n]] = g.interpolate(d).coords[0]
            edges_b[i] = (u, snapped[n], substring(g, 0, d))
            edges_b.append((snapped[n], v, substring(g, d, g.length)))

    len_a, len_b = all_pairs(xy_a, ctrl_edges), all_pairs(xy_b, edges_b)
    scores = []
    for s, t in itertools.combinations(list(xy_a), 2):
        if t not in len_a[s]:
            continue
        la = len_a[s][t]
        if s in snapped and t in snapped and snapped[t] in len_b[snapped[s]]:
            scores.append(1 - min(1, abs(la - len_b[snapped[s]][snapped[t]]) / la))
        else:
            scores.append(0.0)
    return float(np.mean(scores))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_naive_reference_implementation(seed):
    rng = np.random.default_rng(seed)
    segs = _grid_segments(n_streets=3, spacing_px=int(rng.integers(150, 260)), seg_px=90)
    jitter = {}
    moved = [
        [(x + jitter.setdefault((x, y), tuple(rng.normal(0, 2.5, 2)))[0], y + jitter[(x, y)][1]) for x, y in s]
        for s in segs
    ]
    kept = [s for s in moved if rng.random() > 0.15] + [[(100, 150), (40, 20)]]
    G = build_gt_graph(_fc(segs), GSD)
    P = build_gt_graph(_fc(kept), GSD)
    result = apls_detailed(G, P, GSD, max_pairs=None, simplify=False)
    assert result["apls_gt_to_prop"] == pytest.approx(_naive_direction(G, P, GSD), abs=1e-9)
    assert result["apls_prop_to_gt"] == pytest.approx(_naive_direction(P, G, GSD), abs=1e-9)


def test_triu_pair_matches_numpy_ordering():
    for n in (2, 3, 7, 50, 1001):
        q = np.arange(n * (n - 1) // 2)
        i, j = _triu_pair(q, np.full(len(q), n))
        ii, jj = np.triu_indices(n, 1)
        np.testing.assert_array_equal(i, ii)
        np.testing.assert_array_equal(j, jj)


def test_connected_pairs_stay_within_components():
    segs = _straight_segments() + _shift(_straight_segments(), 0.0, 500.0)
    xy, ctrl = _control_graph(_network(build_gt_graph(_fc(segs), GSD), True), GSD, 50.0)
    i, j = _connected_pairs(ctrl, None, 0)
    assert len(i) == 2 * (5 * 4 // 2)
    assert np.all(np.abs(xy[i, 1] - xy[j, 1]) < 1e-9)
    si, sj = _connected_pairs(ctrl, 7, 0)
    assert len(si) == 7
    assert len({(a, b) for a, b in zip(si, sj)}) == 7
    assert np.all(si < sj)
    assert np.all(np.abs(xy[si, 1] - xy[sj, 1]) < 1e-9)


# ─── SpaceNet 8 flood variant ───────────────────────────────────────────────


def _cross(flooded_arm=True):
    arms = [[(200, 200), (200, 0)], [(200, 200), (400, 200)], [(200, 200), (200, 400)], [(200, 200), (0, 200)]]
    flags = [flooded_arm, False, False, False]
    return arms, flags


def test_flood_subgraph_selects_edges():
    arms, flags = _cross()
    G = build_gt_graph(_fc(arms, flags), GSD)
    assert flood_subgraph(G, True).number_of_edges() == 1
    assert flood_subgraph(G, False).number_of_edges() == 3
    assert flood_subgraph(G, False).number_of_nodes() == 4


def test_apls_flood_perfect_prediction():
    arms, flags = _cross()
    gt = _fc(arms, flags)
    result = apls_flood(gt, build_gt_graph(gt, GSD), GSD)
    assert result["apls_traversable"] == pytest.approx(1.0)
    assert result["apls_non_traversable"] == pytest.approx(1.0)
    assert result["apls_flood"] == pytest.approx(1.0)
    assert result["gt_non_traversable_m"] == pytest.approx(100.0)
    assert result["gt_traversable_m"] == pytest.approx(300.0)


def test_apls_flood_missed_flooding_hurts_both_networks():
    arms, flags = _cross()
    pred = build_gt_graph(_fc(arms, [False] * 4), GSD)
    result = apls_flood(_fc(arms, flags), pred, GSD)
    assert result["apls_non_traversable"] == 0.0
    assert 0.0 < result["apls_traversable"] < 1.0
    assert result["pred_non_traversable_m"] == 0.0
    assert result["apls_flood"] == pytest.approx(0.5 * result["apls_traversable"])


def test_apls_flood_without_flooding_scores_flooded_network_as_empty():
    arms, _ = _cross()
    gt = _fc(arms, [False] * 4)
    result = apls_flood(gt, build_gt_graph(gt, GSD), GSD)
    assert result["apls_non_traversable"] == 1.0
    assert result["gt_non_traversable_m"] == 0.0 and result["pred_non_traversable_m"] == 0.0


def test_apls_runs_fast_on_full_tile():
    segs = _grid_segments(n_streets=5, spacing_px=280, seg_px=70, origin=90)
    assert 50 <= len(segs) <= 200
    rng = np.random.default_rng(0)
    flags = list(rng.random(len(segs)) < 0.3)
    gt = _fc(segs, flags)
    jittered = [[(x + rng.normal(0, 1), y + rng.normal(0, 1)) for x, y in s] for s in segs]
    keep = [i for i in range(len(segs)) if i % 17 != 0]
    pred = _fc([jittered[i] for i in keep], [flags[i] for i in keep])

    start = time.perf_counter()
    result = apls_flood(gt, pred, GSD)
    exact = apls(gt, pred, GSD, max_pairs=None)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
    assert 0.0 < exact < 1.0
    assert 0.0 <= result["apls_flood"] <= 1.0
