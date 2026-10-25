"""
Average Path Length Similarity (APLS) for Road Networks.

Length-based APLS (Van Etten et al., 2019; SpaceNet 3/5) and the SpaceNet 8
flood variant that scores the traversable network (flooded roads removed)
and the non-traversable network (flooded roads only) separately.

APLS compares shortest-path lengths between control points of two graphs.
Control points are the graph's nodes plus midpoints injected along every
edge about every 50 m. For one direction A -> B, every control point of A is
snapped to the nearest point of B's edges within a buffer (4 m) and inserted
there as a node; a control point with no edge in range is missing. For each
pair of A's control points connected in A, with path lengths L_A in A and
L_B in B, the pair scores 1 - min(1, |L_A - L_B| / L_A), or 0 when the path
does not exist in B or an endpoint is missing. Pairs not connected in A are
skipped. The directional score is the mean over pairs and the symmetric
APLS is the harmonic mean of APLS(gt -> prop) and APLS(prop -> gt), 0 if
either is 0.

Graph representation shared by ground truth and prediction: an undirected
networkx MultiGraph whose nodes carry pixel coordinates 'x' and 'y' and
whose edges carry 'geometry' (shapely LineString in pixel [x, y]
coordinates), 'length_px', 'length_m' and 'flooded'.
"""

import logging
from collections.abc import Mapping
from typing import NamedTuple, Optional, Union

import networkx as nx
import numpy as np
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from shapely.geometry import LineString, shape
from shapely.ops import substring

logger = logging.getLogger(__name__)

NODE_MERGE_M = 0.05
MIN_EDGE_WEIGHT_M = 1e-9
DIJKSTRA_CHUNK = 256  # sources per Dijkstra call (bounds memory)

RoadsLike = Union[Mapping, nx.Graph]


class _Network(NamedTuple):
    """Graph as arrays: node coordinates and edges with u -> v oriented coordinates."""

    xy: np.ndarray
    u: np.ndarray
    v: np.ndarray
    coords: list
    length_px: np.ndarray


class _WeightedGraph(NamedTuple):
    """Node count and undirected sparse adjacency with lengths in meters."""

    n_nodes: int
    adjacency: csr_matrix


# ─── Graph construction ─────────────────────────────────────────────────────


def _read_lines(roads_fc: Mapping) -> tuple[list[np.ndarray], list[bool], list[int]]:
    """Coordinates, flood flags and feature indices of every non-degenerate line part."""
    lines, flags, index = [], [], []
    for i, feature in enumerate(roads_fc.get("features", [])):
        geom = shape(feature["geometry"])
        parts = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        flooded = bool((feature.get("properties") or {}).get("flooded", False))
        for part in parts:
            if part.geom_type != "LineString" or part.is_empty or part.length <= 0:
                continue
            lines.append(np.asarray(part.coords, dtype=np.float64)[:, :2])
            flags.append(flooded)
            index.append(i)
    return lines, flags, index


def _cluster_points(points: np.ndarray, radius: float) -> np.ndarray:
    """Label points so that points within radius (transitively) share a label."""
    n = len(points)
    pairs = cKDTree(points).query_pairs(max(radius, 0.0), output_type="ndarray")
    adjacency = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(adjacency, directed=False)
    return labels


def build_gt_graph(roads_fc: Mapping, gsd_m: float, snap_px: float = 2.0) -> nx.MultiGraph:
    """
    Build a road graph from reference centerline segments.

    Segment endpoints within snap_px of each other (transitively) become one
    node at their mean position. A node within snap_px of the interior of
    another segment splits that segment there (T-junction). Exact or
    near-duplicate segments (same end nodes, Hausdorff distance <= snap_px)
    are kept once. Segments that cross without sharing a node stay
    unconnected (over/underpasses).

    Args:
        roads_fc: Road FeatureCollection of LineStrings in pixel coordinates
            with properties.flooded.
        gsd_m: Meters per pixel.
        snap_px: Endpoint snapping and T-junction tolerance in pixels.

    Returns:
        MultiGraph with integer nodes ('x', 'y') and edges ('geometry',
        'length_px', 'length_m', 'flooded', 'feature'); graph attribute
        'gsd_m'.
    """
    G = nx.MultiGraph(gsd_m=gsd_m)
    lines, flags, feature_index = _read_lines(roads_fc)
    if not lines:
        return G

    ends = np.array([[c[0], c[-1]] for c in lines]).reshape(-1, 2)
    labels = _cluster_points(ends, snap_px)
    n_nodes = int(labels.max()) + 1
    node_xy = np.zeros((n_nodes, 2))
    np.add.at(node_xy, labels, ends)
    node_xy /= np.bincount(labels, minlength=n_nodes)[:, None]
    start_node, end_node = labels[0::2], labels[1::2]

    geoms = [LineString(c) for c in lines]
    tree = shapely.STRtree(geoms)
    node_i, line_i = tree.query(shapely.points(node_xy), predicate="dwithin", distance=snap_px)
    splits: dict[int, list[tuple[float, int]]] = {}
    for n, l in zip(node_i.tolist(), line_i.tolist()):
        if n in (start_node[l], end_node[l]):
            continue
        d = geoms[l].project(shapely.Point(node_xy[n]))
        if 0.0 < d < geoms[l].length:
            splits.setdefault(l, []).append((d, n))

    for n in range(n_nodes):
        G.add_node(n, x=float(node_xy[n, 0]), y=float(node_xy[n, 1]))

    for l, line in enumerate(geoms):
        cuts = sorted(splits.get(l, []))
        positions = [0.0] + [d for d, _ in cuts] + [line.length]
        nodes = [int(start_node[l])] + [n for _, n in cuts] + [int(end_node[l])]
        for k in range(len(nodes) - 1):
            u, v = nodes[k], nodes[k + 1]
            piece = substring(line, positions[k], positions[k + 1])
            interior = list(piece.coords)[1:-1] if piece.geom_type == "LineString" else []
            geom = LineString([tuple(node_xy[u])] + interior + [tuple(node_xy[v])])
            if u == v and geom.length <= snap_px:
                continue
            if any(
                geom.hausdorff_distance(d["geometry"]) <= snap_px
                for d in G.get_edge_data(u, v, default={}).values()
            ):
                continue
            G.add_edge(
                u, v,
                geometry=geom,
                length_px=float(geom.length),
                length_m=float(geom.length) * gsd_m,
                flooded=flags[l],
                feature=feature_index[l],
            )

    G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
    return G


def prediction_graph_from_postprocess(
    G: nx.Graph,
    gsd_m: Optional[float] = None,
    pixel_offset: float = 0.5,
) -> nx.MultiGraph:
    """
    Convert an inference.postprocess.build_road_graph() graph for scoring.

    Postprocess graphs index skeleton pixels as (row, col); the reference
    vectors use continuous pixel coordinates in which pixel (row, col) spans
    [col, col + 1) x [row, row + 1), so pixel centers are shifted by
    pixel_offset.

    Args:
        G: MultiGraph with (y, x) nodes and edges carrying 'path' (list of
            (y, x)), 'length_m' and 'flooded'.
        gsd_m: Meters per pixel; if None, edge lengths in meters are taken
            from the edges' 'length_m'.
        pixel_offset: Offset added to pixel indices.

    Returns:
        MultiGraph in the build_gt_graph() representation.
    """
    H = nx.MultiGraph(gsd_m=gsd_m)
    index = {}
    for i, n in enumerate(G.nodes):
        index[n] = i
        H.add_node(i, x=float(n[1]) + pixel_offset, y=float(n[0]) + pixel_offset)

    for u, v, data in G.edges(data=True):
        path = [tuple(p) for p in (data.get("path") or [u, v])]
        if len(path) < 2:
            continue
        if path[0] != tuple(u) and path[-1] == tuple(u):
            path = path[::-1]
        geom = LineString([(x + pixel_offset, y + pixel_offset) for y, x in path])
        if geom.length <= 0:
            continue
        H.add_edge(
            index[u], index[v],
            geometry=geom,
            length_px=float(geom.length),
            length_m=float(geom.length) * gsd_m if gsd_m is not None else float(data["length_m"]),
            flooded=bool(data.get("flooded", False)),
        )
    return H


def _as_graph(roads: RoadsLike, gsd_m: float) -> nx.Graph:
    """Accept a road FeatureCollection, a postprocess graph or a scoring graph."""
    if isinstance(roads, Mapping) and "features" in roads:
        return build_gt_graph(roads, gsd_m)
    if isinstance(roads, nx.Graph):
        edge_data = [d for _, _, d in roads.edges(data=True)]
        if edge_data and all("geometry" not in d and "path" in d for d in edge_data):
            return prediction_graph_from_postprocess(roads, gsd_m)
        return roads
    raise TypeError(f"Expected a road FeatureCollection or networkx graph, got {type(roads).__name__}")


def flood_subgraph(G: nx.Graph, flooded: bool) -> nx.MultiGraph:
    """
    Keep only the edges with the given flood state.

    Args:
        G: Graph in the build_gt_graph() representation.
        flooded: True keeps flooded edges (non-traversable network), False
            keeps the others (traversable network).

    Returns:
        New MultiGraph with the selected edges and their end nodes.
    """
    H = nx.MultiGraph(**G.graph)
    for u, v, data in G.edges(data=True):
        if bool(data.get("flooded", False)) == flooded:
            H.add_node(u, **G.nodes[u])
            H.add_node(v, **G.nodes[v])
            H.add_edge(u, v, **data)
    return H


# ─── Internal network arithmetic ────────────────────────────────────────────


def _merge_degree_two(edges: list[list]) -> list[list]:
    """Join the two edges of every pass-through node (degree 2, no self-loop)."""
    alive: list[Optional[list]] = list(edges)
    incident: dict[int, set[int]] = {}
    for e, (u, v, _) in enumerate(edges):
        incident.setdefault(u, set()).add(e)
        incident.setdefault(v, set()).add(e)

    stack = list(incident)
    while stack:
        n = stack.pop()
        eids = incident.get(n, set())
        if len(eids) != 2:
            continue
        e1, e2 = sorted(eids)
        (u1, v1, c1), (u2, v2, c2) = alive[e1], alive[e2]
        if u1 == v1 or u2 == v2:
            continue
        a, ca = (u1, c1) if v1 == n else (v1, c1[::-1])
        b, cb = (v2, c2) if u2 == n else (u2, c2[::-1])
        alive[e1] = alive[e2] = None
        for node in (a, b, n):
            incident[node].difference_update((e1, e2))
        alive.append([a, b, np.vstack([ca, cb[1:]])])
        incident[a].add(len(alive) - 1)
        incident[b].add(len(alive) - 1)
        del incident[n]
        stack.extend((a, b))
    return [e for e in alive if e is not None]


def _network(G: nx.Graph, simplify: bool) -> _Network:
    """Convert a scoring graph into arrays, optionally merging pass-through nodes."""
    if G.number_of_edges() == 0:
        return _Network(np.zeros((0, 2)), np.zeros(0, np.int64), np.zeros(0, np.int64), [], np.zeros(0))
    nodes = list(G.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    try:
        xy = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in nodes], dtype=np.float64).reshape(-1, 2)
    except KeyError as e:
        raise ValueError("Graph nodes need 'x' and 'y' pixel coordinates") from e

    edges = []
    for u, v, data in G.edges(data=True):
        iu, iv = index[u], index[v]
        geom = data.get("geometry")
        if geom is None or geom.is_empty:
            coords = np.array([xy[iu], xy[iv]])
        else:
            coords = np.asarray(geom.coords, dtype=np.float64)[:, :2]
        if np.hypot(*(coords[0] - xy[iu])) > np.hypot(*(coords[0] - xy[iv])):
            coords = coords[::-1]
        edges.append([iu, iv, coords])

    if simplify:
        edges = _merge_degree_two(edges)
    lengths = [float(np.hypot(*np.diff(c, axis=0).T).sum()) for _, _, c in edges]
    edges = [(e, length) for e, length in zip(edges, lengths) if length > 0]

    used = sorted({n for (u, v, _), _ in edges for n in (u, v)})
    remap = {n: i for i, n in enumerate(used)}
    return _Network(
        xy=xy[used] if used else np.zeros((0, 2)),
        u=np.array([remap[u] for (u, _, _), _ in edges], dtype=np.int64),
        v=np.array([remap[v] for (_, v, _), _ in edges], dtype=np.int64),
        coords=[c for (_, _, c), _ in edges],
        length_px=np.array([length for _, length in edges], dtype=np.float64),
    )


def _adjacency(n_nodes: int, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> csr_matrix:
    """Undirected sparse adjacency keeping the shortest of parallel edges."""
    u, v, w = np.asarray(u, dtype=np.int64), np.asarray(v, dtype=np.int64), np.asarray(w, dtype=np.float64)
    keep = u != v
    a, b, w = np.minimum(u, v)[keep], np.maximum(u, v)[keep], np.maximum(w[keep], MIN_EDGE_WEIGHT_M)
    key = a * max(n_nodes, 1) + b
    order = np.lexsort((w, key))
    first = np.ones(len(order), dtype=bool)
    first[1:] = key[order][1:] != key[order][:-1]
    order = order[first]
    return csr_matrix((w[order], (a[order], b[order])), shape=(n_nodes, n_nodes))


def _midpoint_distances(length_m: float, spacing_m: float) -> np.ndarray:
    """Positions (m from the start) of control points injected along an edge."""
    if spacing_m <= 0 or length_m < 0.75 * spacing_m:
        return np.zeros(0)
    if length_m <= spacing_m:
        return np.array([0.5 * length_m])
    n_points = len(np.arange(0.0, length_m, spacing_m)) + 1
    return np.linspace(0.0, length_m, n_points)[1:-1]


def _control_graph(net: _Network, gsd_m: float, spacing_m: float) -> tuple[np.ndarray, _WeightedGraph]:
    """Inject midpoints into every edge; all nodes of the result are control points."""
    xy = [net.xy]
    n_nodes = len(net.xy)
    eu, ev, ew = [], [], []
    for u, v, coords, length_px in zip(net.u, net.v, net.coords, net.length_px):
        length_m = length_px * gsd_m
        dists = _midpoint_distances(length_m, spacing_m)
        chain = [u] + list(range(n_nodes, n_nodes + len(dists))) + [v]
        if len(dists):
            points = shapely.line_interpolate_point(LineString(coords), dists / gsd_m)
            xy.append(shapely.get_coordinates(points))
            n_nodes += len(dists)
        eu.extend(chain[:-1])
        ev.extend(chain[1:])
        ew.extend(np.diff(np.concatenate([[0.0], dists, [length_m]])))
    return np.vstack(xy), _WeightedGraph(n_nodes, _adjacency(n_nodes, eu, ev, ew))


def _snap_points(
    points_xy: np.ndarray,
    net: _Network,
    gsd_m: float,
    buffer_m: float,
) -> tuple[np.ndarray, _WeightedGraph]:
    """
    Insert points into a network at their nearest edge location within buffer_m.

    Returns:
        (node index in the augmented network per point, -1 if missing;
        augmented weighted graph).
    """
    n_points = len(points_xy)
    node_of = np.full(n_points, -1, dtype=np.int64)
    n_nodes = len(net.xy)
    if n_points == 0 or len(net.u) == 0:
        return node_of, _WeightedGraph(n_nodes, _adjacency(n_nodes, net.u, net.v, net.length_px * gsd_m))

    geoms = np.array([LineString(c) for c in net.coords], dtype=object)
    points = shapely.points(points_xy)
    pi, ei = shapely.STRtree(geoms).query_nearest(points, max_distance=buffer_m / gsd_m, all_matches=True)
    order = np.lexsort((ei, pi))
    pi, ei = pi[order], ei[order]
    first = np.ones(len(pi), dtype=bool)
    first[1:] = pi[1:] != pi[:-1]
    pi, ei = pi[first], ei[first]
    along = shapely.line_locate_point(geoms[ei], points[pi])

    inserts: dict[int, list[tuple[float, int]]] = {}
    for p, e, d in zip(pi.tolist(), ei.tolist(), along.tolist()):
        inserts.setdefault(e, []).append((d, p))

    tol_px = NODE_MERGE_M / gsd_m
    eu, ev, ew = [], [], []
    for e, (u, v, length_px) in enumerate(zip(net.u, net.v, net.length_px)):
        chain, pos = [u], [0.0]
        for d, p in sorted(inserts.get(e, [])):
            if d <= tol_px:
                node_of[p] = u
            elif d >= length_px - tol_px:
                node_of[p] = v
            elif len(chain) > 1 and d - pos[-1] <= tol_px:
                node_of[p] = chain[-1]
            else:
                chain.append(n_nodes)
                pos.append(d)
                node_of[p] = n_nodes
                n_nodes += 1
        chain.append(v)
        pos.append(length_px)
        eu.extend(chain[:-1])
        ev.extend(chain[1:])
        ew.extend(np.diff(pos) * gsd_m)
    return node_of, _WeightedGraph(n_nodes, _adjacency(n_nodes, eu, ev, ew))


def _triu_pair(q: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map linear indices q of the strict upper triangle of n x n matrices to (i, j)."""
    q, n = np.asarray(q, dtype=np.int64), np.asarray(n, dtype=np.int64)

    def before(i: np.ndarray) -> np.ndarray:
        return i * (2 * n - i - 1) // 2

    i = np.floor(((2 * n - 1) - np.sqrt(np.maximum((2 * n - 1) ** 2 - 8 * q, 0))) / 2).astype(np.int64)
    i = np.clip(i, 0, np.maximum(n - 2, 0))
    i = np.where(before(i) > q, i - 1, i)
    i = np.where(before(i + 1) <= q, i + 1, i)
    return i, q - before(i) + i + 1


def _sample_distinct(rng: np.random.Generator, total: int, size: int) -> np.ndarray:
    """`size` distinct sorted integers in [0, total) without materializing all of them."""
    if total <= 2 * size:
        return np.sort(rng.choice(total, size=size, replace=False))
    k = np.unique(rng.integers(0, total, size=size))
    while len(k) < size:  # top up the few duplicates
        k = np.unique(np.concatenate([k, rng.integers(0, total, size=size - len(k))]))
    return k


def _connected_pairs(graph: _WeightedGraph, max_pairs: Optional[int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """All, or a uniform sample of, unordered node pairs connected in the graph."""
    if graph.n_nodes < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    _, labels = connected_components(graph.adjacency, directed=False)
    order = np.argsort(labels, kind="stable")
    sizes = np.bincount(labels).astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    n_pairs = sizes * (sizes - 1) // 2
    total = int(n_pairs.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    offsets = np.concatenate([[0], np.cumsum(n_pairs)])
    if max_pairs is None or total <= max_pairs:
        k = np.arange(total, dtype=np.int64)
    else:
        k = _sample_distinct(np.random.default_rng(seed), total, max_pairs)
    comp = np.searchsorted(offsets, k, side="right") - 1
    i, j = _triu_pair(k - offsets[comp], sizes[comp])
    return order[starts[comp] + i], order[starts[comp] + j]


def _path_lengths(graph: _WeightedGraph, src: np.ndarray, dst: np.ndarray,
                  chunk: int = DIJKSTRA_CHUNK) -> np.ndarray:
    """Shortest path lengths src[k] -> dst[k] (inf if disconnected)."""
    out = np.zeros(len(src))
    if len(src) == 0:
        return out
    sources, row = np.unique(src, return_inverse=True)
    # Sources in blocks: memory stays chunk x n_nodes however large (or noisy) the graph is
    for start in range(0, len(sources), chunk):
        dist = dijkstra(graph.adjacency, directed=False, indices=sources[start:start + chunk])
        sel = (row >= start) & (row < start + chunk)
        out[sel] = dist[row[sel] - start, dst[sel]]
    return out


def _directional_apls(
    ctrl_xy: np.ndarray,
    ctrl_graph: _WeightedGraph,
    target: _Network,
    gsd_m: float,
    buffer_m: float,
    max_pairs: Optional[int],
    seed: int,
    min_path_m: float,
    max_sources: Optional[int] = None,
) -> tuple[Optional[float], int]:
    """APLS(A -> B) from A's control graph and network B; None if A has no scorable pair."""
    if max_pairs is None:
        return _directional_apls_streaming(ctrl_xy, ctrl_graph, target, gsd_m, buffer_m, max_sources, seed, min_path_m)
    pi, pj = _connected_pairs(ctrl_graph, max_pairs, seed)
    len_a = _path_lengths(ctrl_graph, pi, pj)
    keep = np.isfinite(len_a) & (len_a >= min_path_m)
    pi, pj, len_a = pi[keep], pj[keep], len_a[keep]
    if len(len_a) == 0:
        return None, 0

    node_of, target_graph = _snap_points(ctrl_xy, target, gsd_m, buffer_m)
    bi, bj = node_of[pi], node_of[pj]
    len_b = np.full(len(len_a), np.inf)
    found = (bi >= 0) & (bj >= 0)
    len_b[found] = _path_lengths(target_graph, bi[found], bj[found])

    diff = np.minimum(1.0, np.abs(len_a - len_b) / len_a)
    diff[~np.isfinite(len_b)] = 1.0
    return float(np.mean(1.0 - diff)), int(len(len_a))


def _directional_apls_streaming(
    ctrl_xy: np.ndarray,
    ctrl_graph: _WeightedGraph,
    target: _Network,
    gsd_m: float,
    buffer_m: float,
    max_sources: Optional[int],
    seed: int,
    min_path_m: float,
) -> tuple[Optional[float], int]:
    """
    Every connected pair, accumulated source by source (memory: a block of
    sources x nodes, never the n^2 pair list). With more than max_sources
    control points, a uniform sample of sources is used instead: each pair is
    then included with the same probability, so the mean stays unbiased.
    Equal to the exact pairwise mean otherwise (pairs are symmetric).

    Returns:
        (APLS, number of unordered pairs scored, or ordered pairs when sampled).
    """
    n = ctrl_graph.n_nodes
    if n < 2:
        return None, 0
    _, labels = connected_components(ctrl_graph.adjacency, directed=False)
    sampled = max_sources is not None and n > max_sources
    sources = np.sort(np.random.default_rng(seed).choice(n, size=max_sources, replace=False)) if sampled \
        else np.arange(n)
    node_of, target_graph = _snap_points(ctrl_xy, target, gsd_m, buffer_m)
    snapped = node_of >= 0

    total, count = 0.0, 0
    for start in range(0, len(sources), DIJKSTRA_CHUNK):
        src = sources[start:start + DIJKSTRA_CHUNK]
        len_a = dijkstra(ctrl_graph.adjacency, directed=False, indices=src)  # (block, n)
        scored = labels[src][:, None] == labels[None, :]
        scored[np.arange(len(src)), src] = False
        scored &= np.isfinite(len_a) & (len_a >= min_path_m)
        if not scored.any():
            continue
        len_b = np.full(len_a.shape, np.inf)
        rows = np.flatnonzero(snapped[src])
        if len(rows) and snapped.any():
            uniq, inv = np.unique(node_of[src[rows]], return_inverse=True)
            dist_b = dijkstra(target_graph.adjacency, directed=False, indices=uniq)
            len_b[np.ix_(rows, np.flatnonzero(snapped))] = dist_b[inv][:, node_of[snapped]]
        a, b = len_a[scored], len_b[scored]
        diff = np.minimum(1.0, np.abs(a - b) / a)
        diff[~np.isfinite(b)] = 1.0
        total += float((1.0 - diff).sum())
        count += int(scored.sum())
    if count == 0:
        return None, 0
    return total / count, count if sampled else count // 2


# ─── Metric ─────────────────────────────────────────────────────────────────


def apls_detailed(
    gt_graph: RoadsLike,
    prop_graph: RoadsLike,
    gsd_m: float,
    buffer_m: float = 4.0,
    node_spacing_m: float = 50.0,
    max_pairs: Optional[int] = 500,
    max_sources: Optional[int] = None,
    seed: int = 0,
    min_path_m: float = 0.001,
    simplify: bool = True,
) -> dict:
    """
    Symmetric APLS with its directional components.

    Args:
        gt_graph: Reference graph (build_gt_graph()), or a road
            FeatureCollection, which is passed through build_gt_graph().
        prop_graph: Proposal graph in the same representation, a
            postprocess.build_road_graph() graph, or a FeatureCollection.
        gsd_m: Meters per pixel.
        buffer_m: Snapping distance for control points, in meters.
        node_spacing_m: Spacing of control points injected along edges.
        max_pairs: Pairs scored per direction; a uniform sample (fixed by
            seed) is drawn from the connected pairs when there are more.
            None scores every connected pair (exact), streamed so memory
            does not grow with the number of pairs.
        max_sources: With max_pairs=None, graphs with more control points
            than this are scored from a uniform sample of this many source
            points (unbiased; bounds the time on huge, noisy graphs).
        seed: Seed of the pair or source sample.
        min_path_m: Pairs whose reference path is shorter are skipped.
        simplify: Merge pass-through (degree-2) nodes before injecting
            control points, so both graphs get control points at junctions,
            dead ends and every node_spacing_m regardless of how finely
            their edges are segmented.

    Returns:
        Dict with 'apls', 'apls_gt_to_prop', 'apls_prop_to_gt' (None when
        that side has no scorable pair), 'n_pairs_gt', 'n_pairs_prop',
        'n_control_gt', 'n_control_prop', 'gt_length_m' and 'prop_length_m'.
        'apls' is 1.0 when both graphs are empty and 0.0 when exactly one is.
    """
    if gsd_m <= 0:
        raise ValueError(f"gsd_m must be positive, got {gsd_m}")
    gt = _network(_as_graph(gt_graph, gsd_m), simplify)
    prop = _network(_as_graph(prop_graph, gsd_m), simplify)
    gt_xy, gt_ctrl = _control_graph(gt, gsd_m, node_spacing_m)
    prop_xy, prop_ctrl = _control_graph(prop, gsd_m, node_spacing_m)

    kw = dict(gsd_m=gsd_m, buffer_m=buffer_m, max_pairs=max_pairs, seed=seed, min_path_m=min_path_m,
              max_sources=max_sources)
    gt_to_prop, n_gt = _directional_apls(gt_xy, gt_ctrl, prop, **kw)
    prop_to_gt, n_prop = _directional_apls(prop_xy, prop_ctrl, gt, **kw)

    if gt_to_prop is None and prop_to_gt is None:
        score = 1.0
    elif gt_to_prop is None or prop_to_gt is None or gt_to_prop <= 0 or prop_to_gt <= 0:
        score = 0.0
    else:
        score = 2.0 * gt_to_prop * prop_to_gt / (gt_to_prop + prop_to_gt)

    return {
        "apls": float(score),
        "apls_gt_to_prop": gt_to_prop,
        "apls_prop_to_gt": prop_to_gt,
        "n_pairs_gt": n_gt,
        "n_pairs_prop": n_prop,
        "n_control_gt": int(gt_ctrl.n_nodes),
        "n_control_prop": int(prop_ctrl.n_nodes),
        "gt_length_m": float(gt.length_px.sum() * gsd_m),
        "prop_length_m": float(prop.length_px.sum() * gsd_m),
    }


def apls(
    gt_graph: RoadsLike,
    prop_graph: RoadsLike,
    gsd_m: float,
    buffer_m: float = 4.0,
    node_spacing_m: float = 50.0,
    max_pairs: Optional[int] = 500,
    max_sources: Optional[int] = None,
    seed: int = 0,
    min_path_m: float = 0.001,
    simplify: bool = True,
) -> float:
    """
    Symmetric length-based APLS: harmonic mean of APLS(gt->prop) and APLS(prop->gt).

    Args:
        gt_graph: Reference graph or road FeatureCollection.
        prop_graph: Proposal graph, postprocess graph or FeatureCollection.
        gsd_m: Meters per pixel.
        buffer_m: Snapping distance for control points, in meters.
        node_spacing_m: Spacing of control points injected along edges.
        max_pairs: Pairs scored per direction (None = all connected pairs).
        max_sources: Source-point cap when max_pairs is None (see apls_detailed).
        seed: Seed of the pair or source sample.
        min_path_m: Pairs whose reference path is shorter are skipped.
        simplify: Merge pass-through nodes before injecting control points.

    Returns:
        APLS in [0, 1]; 1.0 when both graphs are empty, 0.0 when one is.
    """
    return apls_detailed(
        gt_graph, prop_graph, gsd_m,
        buffer_m=buffer_m,
        node_spacing_m=node_spacing_m,
        max_pairs=max_pairs,
        max_sources=max_sources,
        seed=seed,
        min_path_m=min_path_m,
        simplify=simplify,
    )["apls"]


def apls_flood(
    gt_roads_fc: RoadsLike,
    pred_graph: RoadsLike,
    gsd_m: float,
    **apls_kwargs,
) -> dict:
    """
    SpaceNet 8 road scoring: APLS of the traversable and flooded networks.

    The traversable network keeps the non-flooded edges of each graph, the
    non-traversable network only the flooded ones; each pair is scored with
    apls() and the two scores are averaged. A network empty in both graphs
    scores 1.0 by construction, so the returned lengths allow tiles without
    any flooded road to be excluded when aggregating.

    Args:
        gt_roads_fc: Reference road FeatureCollection (or a graph from
            build_gt_graph()).
        pred_graph: postprocess.build_road_graph() graph, a converted graph
            or a road FeatureCollection.
        gsd_m: Meters per pixel.
        **apls_kwargs: Passed to apls() (buffer_m, node_spacing_m,
            max_pairs, max_sources, seed, min_path_m, simplify).

    Returns:
        Dict with 'apls_traversable', 'apls_non_traversable', 'apls_flood'
        (their mean) and the total edge length in meters of each network:
        'gt_traversable_m', 'gt_non_traversable_m', 'pred_traversable_m',
        'pred_non_traversable_m'.
    """
    gt = _as_graph(gt_roads_fc, gsd_m)
    pred = _as_graph(pred_graph, gsd_m)
    result = {}
    for name, flooded in (("traversable", False), ("non_traversable", True)):
        gt_sub, pred_sub = flood_subgraph(gt, flooded), flood_subgraph(pred, flooded)
        result[f"apls_{name}"] = apls(gt_sub, pred_sub, gsd_m, **apls_kwargs)
        result[f"gt_{name}_m"] = _total_length_m(gt_sub, gsd_m)
        result[f"pred_{name}_m"] = _total_length_m(pred_sub, gsd_m)
    result["apls_flood"] = 0.5 * (result["apls_traversable"] + result["apls_non_traversable"])
    return result


def _total_length_m(G: nx.Graph, gsd_m: float) -> float:
    """Sum of edge geometry lengths in meters."""
    return float(sum(d["geometry"].length for _, _, d in G.edges(data=True) if d.get("geometry") is not None) * gsd_m)
