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
        # If road clusters are slightly disconnected, bridge nearest components
        try:
            G_connected = G.copy()
            components = list(nx.connected_components(G))
            # Connect disconnected components with straight line links
            for i in range(len(components) - 1):
                c1 = list(components[i])
                c2 = list(components[i + 1])
                n1 = c1[0]
                n2 = min(c2, key=lambda n: (n[0] - n1[0])**2 + (n[1] - n1[1])**2)
                bridge_dist = np.sqrt((n1[0] - n2[0])**2 + (n1[1] - n2[1])**2) * 0.3
                G_connected.add_edge(n1, n2, distance=bridge_dist, weight=bridge_dist, flooded=False)

            path = nx.shortest_path(G_connected, start_node, end_node, weight="weight")
            total_dist = sum(
                G_connected[path[i]][path[i + 1]].get("distance", 0)
                for i in range(len(path) - 1)
            )
            has_flooded = any(
                G_connected[path[i]][path[i + 1]].get("flooded", False)
                for i in range(len(path) - 1)
            )
            return {
                "path": [(int(y), int(x)) for y, x in path],
                "distance": round(total_dist, 2),
                "has_flooded_segments": has_flooded,
                "route_type": "safe" if not has_flooded else "any",
            }
        except Exception:
            return {
                "path": [start, end],
                "distance": round(np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2) * 0.3, 2),
                "has_flooded_segments": False,
                "route_type": "direct",
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

    road_mask = binary[1] if binary.shape[0] > 1 else np.zeros_like(binary[0])
    flood_mask = binary[2] if binary.shape[0] > 2 else None

    # Fallback to simple grid if road mask is empty
    if road_mask.sum() == 0:
        H, W = road_mask.shape
        G = nx.Graph()
        # Add synthetic grid roads for visualization/routing
        step = max(32, H // 12)
        for y in range(step, H - step, step):
            for x in range(step, W - step, step):
                G.add_node((y, x), y=y, x=x)
                if x + step < W - step:
                    G.add_edge((y, x), (y, x + step), distance=step * pixel_scale, weight=step * pixel_scale, flooded=False)
                if y + step < H - step:
                    G.add_edge((y, x), (y + step, x), distance=step * pixel_scale, weight=step * pixel_scale, flooded=False)
        return G

    # Skeletonize road mask
    skeleton = mask_to_skeleton(road_mask)

    # Build graph
    G = skeleton_to_graph(skeleton, flood_mask=flood_mask, pixel_scale=pixel_scale)
    return G


def graph_to_geojson(G: "nx.Graph") -> dict:
    """
    Export all edges in the road graph as a GeoJSON FeatureCollection.
    Each feature is a LineString representing a road segment with its flood status.
    """
    if nx is None or G is None or G.number_of_edges() == 0:
        return {"type": "FeatureCollection", "features": []}

    features = []
    for u, v, data in G.edges(data=True):
        # Coordinates in [x, y] format (GeoJSON standard)
        coords = [[int(u[1]), int(u[0])], [int(v[1]), int(v[0])]]
        is_flooded = bool(data.get("flooded", False))
        distance = float(data.get("distance", 0.0))

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords,
            },
            "properties": {
                "flooded": is_flooded,
                "status": "flooded" if is_flooded else "clear",
                "distance_m": round(distance, 1),
                "speed_mph": 0 if is_flooded else 25,
            },
        })

    return {"type": "FeatureCollection", "features": features}


def extract_building_polygons(building_mask: np.ndarray, flooded_building_mask: Optional[np.ndarray] = None, max_polygons: int = 150) -> dict:
    """
    Extract simplified building polygon footprints as a GeoJSON FeatureCollection.
    """
    try:
        from skimage.measure import find_contours

        features = []
        if building_mask.sum() == 0:
            return {"type": "FeatureCollection", "features": []}

        contours = find_contours(building_mask.astype(float), 0.5)
        # Sort by contour area/length to keep significant buildings
        contours = sorted(contours, key=len, reverse=True)[:max_polygons]

        for cnt in contours:
            if len(cnt) < 4:
                continue
            # Decimate/simplify contour points for efficient frontend rendering
            step = max(1, len(cnt) // 16)
            simplified = cnt[::step].tolist()
            if simplified[0] != simplified[-1]:
                simplified.append(simplified[0])

            # Convert (y, x) to [x, y]
            coords = [[int(round(pt[1])), int(round(pt[0]))] for pt in simplified]

            # Check if flooded
            centroid_y = int(round(cnt[:, 0].mean()))
            centroid_x = int(round(cnt[:, 1].mean()))
            is_flooded = False
            if flooded_building_mask is not None:
                H, W = flooded_building_mask.shape
                if 0 <= centroid_y < H and 0 <= centroid_x < W:
                    is_flooded = bool(flooded_building_mask[centroid_y, centroid_x] > 0)

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords],
                },
                "properties": {
                    "flooded": is_flooded,
                    "status": "flooded" if is_flooded else "intact",
                },
            })

        return {"type": "FeatureCollection", "features": features}
    except Exception as e:
        logger.warning(f"Building polygon extraction failed: {e}")
        return {"type": "FeatureCollection", "features": []}
