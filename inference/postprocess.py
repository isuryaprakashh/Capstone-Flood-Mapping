"""
Vectorization of Flood Predictions.

Converts predicted raster masks into the object-level outputs SpaceNet 8 is
scored on: building footprint polygons and a road network graph, each with a
flooded / not-flooded attribute decided by majority vote over the object's
pixels (as in the SpaceNet 8 baseline post-processing).

All coordinates are in pixel space of the pre-event tile ([x, y] order in
GeoJSON); lengths and areas use the tile's ground sampling distance.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import networkx as nx
except ImportError:
    nx = None
    logger.warning("NetworkX not installed. Road graph features disabled.")


# ─── Buildings ──────────────────────────────────────────────────────────────


def extract_buildings(
    building_mask: np.ndarray,
    flooded_mask: Optional[np.ndarray] = None,
    gsd_m: float = 0.5,
    min_area_m2: float = 5.0,
    flood_fraction: float = 0.5,
    simplify_px: float = 1.0,
) -> dict:
    """
    Polygonize building instances and attribute their flood status.

    Args:
        building_mask: Binary building mask (H, W).
        flooded_mask: Binary flooded-building mask (H, W).
        gsd_m: Meters per pixel.
        min_area_m2: Drop polygons smaller than this.
        flood_fraction: Fraction of an instance's pixels that must be
            flooded for the building to be flooded (majority vote).
        simplify_px: Polygon simplification tolerance in pixels.

    Returns:
        GeoJSON FeatureCollection; each feature has 'flooded' and 'area_m2'.
    """
    from rasterio.features import shapes
    from scipy.ndimage import binary_opening
    from shapely.geometry import mapping, shape
    from skimage.measure import label

    mask = binary_opening(building_mask > 0, structure=np.ones((3, 3), dtype=bool))
    labels = label(mask, connectivity=1).astype(np.int32)
    if labels.max() == 0:
        return {"type": "FeatureCollection", "features": []}

    # Per-instance flooded fraction via bincount (index = instance id)
    counts = np.bincount(labels.ravel())
    if flooded_mask is not None:
        flooded_px = np.bincount(labels.ravel(), weights=(flooded_mask > 0).ravel().astype(np.float64))
        frac = flooded_px / np.maximum(counts, 1)
    else:
        frac = np.zeros_like(counts, dtype=np.float64)

    min_px = min_area_m2 / max(gsd_m, 1e-6) ** 2
    features = []
    for geom, value in shapes(labels, mask=labels > 0, connectivity=4):
        poly = shape(geom)
        if poly.area < min_px:
            continue
        poly = poly.simplify(simplify_px, preserve_topology=True)
        flooded = bool(frac[int(value)] >= flood_fraction)
        features.append({
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {
                "flooded": flooded,
                "status": "flooded" if flooded else "intact",
                "area_m2": round(float(poly.area) * gsd_m ** 2, 1),
            },
        })

    return {"type": "FeatureCollection", "features": features}


# ─── Roads ──────────────────────────────────────────────────────────────────


def build_road_graph(
    road_mask: np.ndarray,
    flooded_mask: Optional[np.ndarray] = None,
    gsd_m: float = 0.5,
    flood_fraction: float = 0.3,
    flood_dilation_px: int = 5,
    min_component_m: float = 10.0,
) -> "nx.MultiGraph":
    """
    Skeletonize a road mask into a graph with flood-attributed edges.

    Edges are chains of skeleton pixels between junctions/endpoints. An edge
    is flooded when more than ``flood_fraction`` of its pixels fall on the
    (dilated) flooded-road mask; dilation tolerates small pre/post offsets.

    Args:
        road_mask: Binary road mask (H, W).
        flooded_mask: Binary flooded-road mask (H, W).
        gsd_m: Meters per pixel.
        flood_fraction: Majority-vote threshold for flooded edges.
        flood_dilation_px: Square dilation applied to the flooded mask.
        min_component_m: Drop connected components shorter than this.

    Returns:
        NetworkX MultiGraph; nodes are (y, x) pixels, edges carry 'path'
        (list of (y, x)), 'length_m' and 'flooded'.
    """
    if nx is None:
        raise ImportError("NetworkX required for road graph extraction")

    from scipy.ndimage import binary_dilation
    from skimage.morphology import skeletonize

    skeleton = skeletonize(road_mask > 0)
    flooded = None
    if flooded_mask is not None and flooded_mask.any():
        flooded = binary_dilation(
            flooded_mask > 0, structure=np.ones((flood_dilation_px, flood_dilation_px), dtype=bool)
        )

    pixel_graph = _skeleton_pixel_graph(skeleton)
    G = _contract_chains(pixel_graph, flooded, gsd_m, flood_fraction)

    # Remove short isolated fragments (skeleton noise)
    for component in list(nx.connected_components(G)):
        sub = G.subgraph(component)
        if sum(d["length_m"] for _, _, d in sub.edges(data=True)) < min_component_m:
            G.remove_nodes_from(component)

    logger.info(f"Road graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def _skeleton_pixel_graph(skeleton: np.ndarray) -> "nx.Graph":
    """8-connected graph over skeleton pixels."""
    G = nx.Graph()
    ys, xs = np.nonzero(skeleton)
    pixels = set(zip(ys.tolist(), xs.tolist()))
    G.add_nodes_from(pixels)
    for y, x in pixels:
        for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
            if (y + dy, x + dx) in pixels:
                G.add_edge((y, x), (y + dy, x + dx))
    return G


def _contract_chains(
    G: "nx.Graph",
    flooded: Optional[np.ndarray],
    gsd_m: float,
    flood_fraction: float,
) -> "nx.MultiGraph":
    """
    Contract skeleton pixel chains into edges between junctions/endpoints.

    Adjacent junction pixels (the small pixel clusters skeletonization leaves
    at intersections) are merged into one node, and junction-free pass-through
    nodes are merged away, so a "+" intersection yields exactly four edges.
    Chains are also split where the (dilated) flood mask starts or ends, so
    flood attribution follows flood boundaries instead of one majority vote
    over a whole junction-to-junction road. A MultiGraph keeps parallel roads
    between the same two junctions.
    """
    junctions = [n for n in G.nodes if G.degree(n) > 2]
    rep = {}
    for cluster in nx.connected_components(G.subgraph(junctions)):
        cluster = list(cluster)
        cy, cx = np.mean(cluster, axis=0)
        center = min(cluster, key=lambda p: (p[0] - cy) ** 2 + (p[1] - cx) ** 2)
        for p in cluster:
            rep[p] = center
    for n in G.nodes:
        if G.degree(n) < 2:
            rep[n] = n
    # Split chains where flood status changes, so each edge gets a uniform flood label
    if flooded is not None:
        for n in G.nodes:
            if n not in rep and flooded[n] and any(not flooded[m] for m in G.neighbors(n)):
                rep[n] = n
    # Pure cycles have no junctions or endpoints: anchor one node per cycle
    for component in nx.connected_components(G):
        if not any(p in rep for p in component):
            anchor = next(iter(component))
            rep[anchor] = anchor

    simple = nx.MultiGraph()
    simple.add_nodes_from(set(rep.values()))
    visited = set()

    for start in rep:
        for neighbor in G.neighbors(start):
            if rep.get(neighbor) == rep[start] or frozenset((start, neighbor)) in visited:
                continue  # inside a junction cluster, or already traced
            visited.add(frozenset((start, neighbor)))
            chain = [start, neighbor]
            while chain[-1] not in rep:
                nxt = [n for n in G.neighbors(chain[-1]) if n != chain[-2]]
                if not nxt or frozenset((chain[-1], nxt[0])) in visited:
                    break
                visited.add(frozenset((chain[-1], nxt[0])))
                chain.append(nxt[0])

            u, v = rep[start], rep.get(chain[-1], chain[-1])
            if u == v and len(chain) < 4:
                continue
            path = ([u] if u != chain[0] else []) + chain + ([v] if v != chain[-1] else [])
            simple.add_edge(u, v, **_edge_attrs(path, chain, flooded, gsd_m, flood_fraction))

    _merge_pass_through_nodes(simple, flood_fraction)
    simple.remove_nodes_from([n for n in list(simple.nodes) if simple.degree(n) == 0])
    return simple


def _edge_attrs(path, chain, flooded, gsd_m, flood_fraction) -> dict:
    """Geometry, length and majority-vote flood status of one edge."""
    pts = np.array(path)
    px = np.array(chain)
    n_flooded = int(flooded[px[:, 0], px[:, 1]].sum()) if flooded is not None else 0
    return {
        "path": [(int(y), int(x)) for y, x in path],
        "length_m": float(np.hypot(*np.diff(pts, axis=0).T).sum()) * gsd_m,
        "n_px": len(chain),
        "n_flooded_px": n_flooded,
        "flooded": n_flooded / max(len(chain), 1) > flood_fraction,
    }


def _merge_pass_through_nodes(G: "nx.MultiGraph", flood_fraction: float):
    """Join the two edges of every degree-2 node into a single edge."""
    changed = True
    while changed:
        changed = False
        for n in list(G.nodes):
            edges = list(G.edges(n, keys=True, data=True))
            if G.degree(n) != 2 or len(edges) != 2:
                continue  # junction, endpoint or self-loop
            (_, a, ka, da), (_, b, kb, db) = edges
            if a == n or b == n or da["flooded"] != db["flooded"]:
                continue  # self-loop, or a flood boundary that must stay a node
            pa = da["path"] if da["path"][-1] == n else da["path"][::-1]  # a ... n
            pb = db["path"] if db["path"][0] == n else db["path"][::-1]   # n ... b
            n_px = da["n_px"] + db["n_px"]
            n_flooded = da["n_flooded_px"] + db["n_flooded_px"]
            G.remove_edge(n, a, ka)
            G.remove_edge(n, b, kb)
            G.remove_node(n)
            G.add_edge(
                a, b,
                path=pa + pb[1:],
                length_m=da["length_m"] + db["length_m"],
                n_px=n_px,
                n_flooded_px=n_flooded,
                flooded=n_flooded / max(n_px, 1) > flood_fraction,
            )
            changed = True


def road_graph_to_geojson(G: "nx.Graph", simplify_px: float = 1.0) -> dict:
    """
    Export road graph edges as GeoJSON LineStrings following the road shape.

    Args:
        G: Graph from build_road_graph().
        simplify_px: Line simplification tolerance in pixels.

    Returns:
        GeoJSON FeatureCollection; each feature has 'flooded' and 'length_m'.
    """
    from shapely.geometry import LineString

    features = []
    if G is None:
        return {"type": "FeatureCollection", "features": features}

    for _, _, data in G.edges(data=True):
        path = data.get("path") or []
        if len(path) < 2:
            continue
        line = LineString([(x, y) for y, x in path]).simplify(simplify_px)
        flooded = bool(data.get("flooded", False))
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [list(c) for c in line.coords]},
            "properties": {
                "flooded": flooded,
                "status": "flooded" if flooded else "passable",
                "length_m": round(float(data.get("length_m", 0.0)), 1),
            },
        })

    return {"type": "FeatureCollection", "features": features}


# ─── Summary ────────────────────────────────────────────────────────────────


def summarize(buildings: dict, roads: dict) -> dict:
    """
    Object-level damage summary of building and road FeatureCollections.

    Returns:
        Dictionary with building counts, road lengths (km) and flooded shares.
    """
    b = buildings.get("features", [])
    r = roads.get("features", [])
    n_flooded = sum(1 for f in b if f["properties"].get("flooded"))
    road_m = sum(f["properties"].get("length_m", 0.0) for f in r)
    flooded_m = sum(f["properties"].get("length_m", 0.0) for f in r if f["properties"].get("flooded"))

    return {
        "buildings": len(b),
        "flooded_buildings": n_flooded,
        "flooded_building_pct": round(100.0 * n_flooded / len(b), 1) if b else 0.0,
        "road_km": round(road_m / 1000.0, 3),
        "flooded_road_km": round(flooded_m / 1000.0, 3),
        "flooded_road_pct": round(100.0 * flooded_m / road_m, 1) if road_m else 0.0,
    }


def vectorize_prediction(prediction: dict, gsd_m: float = 0.5) -> dict:
    """
    Vectorize a FloodPredictor output into buildings, roads and a summary.

    Args:
        prediction: Dict with 'binary_mask' (5, H, W) in channel order
            [building, road, flood, flooded_building, flooded_road].
        gsd_m: Meters per pixel of the pre-event tile.

    Returns:
        Dictionary with 'buildings', 'roads' (FeatureCollections), 'graph'
        and 'summary'.
    """
    binary = prediction["binary_mask"]
    buildings = extract_buildings(binary[0], binary[3] if binary.shape[0] > 3 else None, gsd_m=gsd_m)
    graph = build_road_graph(binary[1], binary[4] if binary.shape[0] > 4 else None, gsd_m=gsd_m)
    roads = road_graph_to_geojson(graph)
    return {
        "buildings": buildings,
        "roads": roads,
        "graph": graph,
        "summary": summarize(buildings, roads),
    }
