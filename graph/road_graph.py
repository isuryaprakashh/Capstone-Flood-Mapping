"""
Road Graph Connectivity Engine for Flood-Safe Routing.

Builds a NetworkX road network graph from predicted road segmentation masks,
marks flooded edges, and computes shortest safe paths that avoid flooded roads.

Used by the API's /route endpoint to help emergency responders
navigate around flood-damaged road segments.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import networkx as nx
except ImportError:
    nx = None
    logger.warning("NetworkX not installed. Road graph features disabled.")

try:
    from skimage.morphology import skeletonize
    from skimage.measure import label as sk_label
except ImportError:
    skeletonize = None
    logger.warning("scikit-image not installed. Skeletonization disabled.")


def mask_to_skeleton(road_mask: np.ndarray) -> np.ndarray:
    """
    Skeletonize a binary road mask to extract 1-pixel-wide centerlines.

    Args:
        road_mask: Binary road mask (H, W), values 0 or 1.

    Returns:
        Skeletonized mask (H, W) with 1-pixel road centerlines.
    """
    if skeletonize is None:
        raise ImportError("scikit-image required for skeletonization")

    # Ensure binary
    binary = (road_mask > 0).astype(np.uint8)
    skeleton = skeletonize(binary).astype(np.uint8)

    return skeleton


def skeleton_to_graph(
    skeleton: np.ndarray,
    flood_mask: Optional[np.ndarray] = None,
    pixel_scale: float = 1.0,
) -> "nx.Graph":
    """
    Convert a skeletonized road mask into a NetworkX graph.

    Each skeleton pixel becomes a node. Adjacent skeleton pixels
    are connected by edges. Edge weights represent road length.
    Flooded edges are marked with a penalty weight.

    Args:
        skeleton: Skeletonized binary mask (H, W).
        flood_mask: Optional flood mask (H, W) for marking flooded edges.
        pixel_scale: Meters per pixel for distance calculation.

    Returns:
        NetworkX Graph with road network topology.
    """
    if nx is None:
        raise ImportError("NetworkX required for graph operations")

    G = nx.Graph()

    # Find all skeleton pixels
    ys, xs = np.where(skeleton > 0)

    if len(ys) == 0:
        return G

    # Create nodes
    for y, x in zip(ys, xs):
        G.add_node((int(y), int(x)), y=int(y), x=int(x))

    # 8-connectivity: connect adjacent pixels
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
               (0, 1), (1, -1), (1, 0), (1, 1)]

    for y, x in zip(ys, xs):
        for dy, dx in offsets:
            ny, nx_coord = y + dy, x + dx
            if (ny, nx_coord) in G:
                # Diagonal distance = sqrt(2), cardinal = 1
                dist = np.sqrt(dy ** 2 + dx ** 2) * pixel_scale

                # Check if edge crosses flooded region
                is_flooded = False
                if flood_mask is not None:
                    mid_y = (y + ny) // 2
                    mid_x = (x + nx_coord) // 2
                    is_flooded = bool(
                        flood_mask[y, x] > 0
                        or flood_mask[ny, nx_coord] > 0
                        or flood_mask[mid_y, mid_x] > 0
                    )

                # Higher weight for flooded roads to discourage routing
                weight = dist if not is_flooded else dist * 1000.0

                G.add_edge(
                    (int(y), int(x)),
                    (int(ny), int(nx_coord)),
                    weight=weight,
                    distance=dist,
                    flooded=is_flooded,
                )

    # Simplify: contract degree-2 nodes into longer edges
    G = _simplify_graph(G)

    logger.info(
        f"Built road graph: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )

    return G


def _simplify_graph(G: "nx.Graph") -> "nx.Graph":
    """
    Simplify graph by contracting chains of degree-2 nodes.

    This reduces the graph size dramatically while preserving topology.
    """
    if nx is None or G.number_of_nodes() == 0:
        return G

    # Find nodes that are NOT degree-2 (junctions, endpoints)
    keep_nodes = set()
    for node in G.nodes():
        if G.degree(node) != 2:
            keep_nodes.add(node)

    if not keep_nodes:
        # All nodes are degree-2 → it's a simple cycle, keep a few
        nodes = list(G.nodes())
        if len(nodes) > 10:
            keep_nodes = set(nodes[::max(1, len(nodes) // 10)])
        else:
            return G

    # Build simplified graph
    simple = nx.Graph()

    for node in keep_nodes:
        simple.add_node(node, **G.nodes[node])

    # Trace paths between junction nodes
    visited_edges = set()
    for start in keep_nodes:
        for neighbor in G.neighbors(start):
            edge_key = (min(start, neighbor), max(start, neighbor))
            if edge_key in visited_edges:
                continue

            # Walk along the chain
            path = [start, neighbor]
            total_dist = G[start][neighbor].get("distance", 1.0)
            total_weight = G[start][neighbor].get("weight", 1.0)
            any_flooded = G[start][neighbor].get("flooded", False)

            visited_edges.add(edge_key)
            current = neighbor

            while current not in keep_nodes:
                neighbors = [n for n in G.neighbors(current) if n != path[-2]]
                if not neighbors:
                    break
                nxt = neighbors[0]
                edge_key = (min(current, nxt), max(current, nxt))
                if edge_key in visited_edges:
                    break
                visited_edges.add(edge_key)

                total_dist += G[current][nxt].get("distance", 1.0)
                total_weight += G[current][nxt].get("weight", 1.0)
                any_flooded = any_flooded or G[current][nxt].get("flooded", False)

                path.append(nxt)
                current = nxt

            end = path[-1]
            if end in keep_nodes and start != end:
                simple.add_edge(
                    start,
                    end,
                    weight=total_weight,
                    distance=total_dist,
                    flooded=any_flooded,
                    path_length=len(path),
                )

    return simple


def find_safe_route(
    G: "nx.Graph",
    start: tuple[int, int],
    end: tuple[int, int],
    avoid_flooded: bool = True,
) -> dict:
    """
    Find the shortest safe route between two points.

    If avoid_flooded is True, flooded edges have very high weight
    so Dijkstra naturally avoids them. Falls back to any-path
    if no flood-free route exists.

    Args:
        G: Road network graph.
        start: Start node (y, x) pixel coordinates.
        end: End node (y, x) pixel coordinates.
        avoid_flooded: Whether to avoid flooded road segments.

    Returns:
        Dictionary with route info:
          - 'path': List of (y, x) node coordinates
          - 'distance': Total route distance
          - 'has_flooded_segments': Whether route uses any flooded roads
          - 'route_type': 'safe' or 'any'
    """
    if nx is None:
        raise ImportError("NetworkX required for routing")

    if G.number_of_nodes() == 0:
        return {"path": [], "distance": 0, "has_flooded_segments": False, "route_type": "empty"}

    # Snap start/end to nearest graph nodes
    start_node = _find_nearest_node(G, start)
    end_node = _find_nearest_node(G, end)

    if start_node is None or end_node is None:
        return {"path": [], "distance": 0, "has_flooded_segments": False, "route_type": "no_node"}

    try:
        # Shortest path using weights (flooded edges have high weight)
        path = nx.shortest_path(G, start_node, end_node, weight="weight")
        total_dist = sum(
            G[path[i]][path[i + 1]].get("distance", 0)
            for i in range(len(path) - 1)
        )
        has_flooded = any(
            G[path[i]][path[i + 1]].get("flooded", False)
            for i in range(len(path) - 1)
        )

        route_type = "any" if has_flooded and avoid_flooded else "safe"

        return {
            "path": [(int(y), int(x)) for y, x in path],
            "distance": round(total_dist, 2),
            "has_flooded_segments": has_flooded,
            "route_type": route_type,
        }

    except nx.NetworkXNoPath:
        return {
            "path": [],
            "distance": 0,
            "has_flooded_segments": False,
            "route_type": "no_path",
        }


def _find_nearest_node(
    G: "nx.Graph", point: tuple[int, int]
) -> Optional[tuple[int, int]]:
    """Find the nearest graph node to a given point."""
    if G.number_of_nodes() == 0:
        return None

    best_node = None
    best_dist = float("inf")

    for node in G.nodes():
        dy = node[0] - point[0]
        dx = node[1] - point[1]
        dist = dy * dy + dx * dx
        if dist < best_dist:
            best_dist = dist
            best_node = node

    return best_node


def route_to_geojson(route: dict) -> dict:
    """
    Convert a route result to GeoJSON LineString for frontend rendering.

    Note: This returns pixel coordinates. For geographic coordinates,
    you'd need to apply the inverse geotransform.

    Args:
        route: Route dictionary from find_safe_route().

    Returns:
        GeoJSON Feature dictionary.
    """
    path = route.get("path", [])

    if not path:
        return {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": []},
            "properties": route,
        }

    # Convert (y, x) pixel coords to [x, y] GeoJSON convention
    coordinates = [[x, y] for y, x in path]

    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coordinates},
        "properties": {
            "distance": route.get("distance", 0),
            "has_flooded_segments": route.get("has_flooded_segments", False),
            "route_type": route.get("route_type", "unknown"),
        },
    }


def build_road_graph_from_prediction(
    prediction: dict,
    pixel_scale: float = 0.3,
) -> "nx.Graph":
    """
    Build a road graph from model prediction output.

    Args:
        prediction: Output from FloodPredictor.predict().
        pixel_scale: Meters per pixel (SpaceNet 8 ≈ 0.3m/px).

    Returns:
        NetworkX road graph with flood annotations.
    """
    binary = prediction["binary_mask"]

    road_mask = binary[1]  # Channel 1: roads
    flood_mask = binary[2] if binary.shape[0] > 2 else None  # Channel 2: flood

    # Skeletonize road mask
    skeleton = mask_to_skeleton(road_mask)

    # Build graph
    G = skeleton_to_graph(skeleton, flood_mask=flood_mask, pixel_scale=pixel_scale)

    return G
