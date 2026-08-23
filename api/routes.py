"""
API Routes for SpaceNet 8 Flood Mapping.

Endpoints:
  POST /api/predict  — Upload pre/post images → flood segmentation
  POST /api/route    — Start/end coords → safe route avoiding floods
  GET  /api/health   — Health check
  GET  /api/sample   — Sample prediction for demo
"""

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "SpaceNet 8 Flood Mapping API",
        "version": "1.0.0",
        "stub_mode": os.getenv("USE_STUB", "true").lower() == "true",
    }


@router.post("/predict")
async def predict_flood(
    request: Request,
    pre_image: UploadFile = File(..., description="Pre-event GeoTIFF image"),
    post_image: UploadFile = File(..., description="Post-event GeoTIFF image"),
):
    """
    Run flood damage prediction on a pre/post satellite image pair.

    Returns multi-class segmentation results:
      - Building footprints
      - Road network
      - Flood extent
      - Flooded buildings
      - Flooded roads

    Returns:
        JSON with prediction masks and statistics.
    """
    try:
        predictor = getattr(request.app.state, "predictor", None)

        if predictor is not None:
            # Real model inference
            with tempfile.TemporaryDirectory() as tmpdir:
                pre_path = Path(tmpdir) / "pre.tif"
                post_path = Path(tmpdir) / "post.tif"

                pre_content = await pre_image.read()
                post_content = await post_image.read()

                pre_path.write_bytes(pre_content)
                post_path.write_bytes(post_content)

                result = predictor.predict_from_files(pre_path, post_path)
        else:
            # Stub mode
            from inference.predict import create_stub_prediction

            result = create_stub_prediction(512, 512)

        # Convert masks to serializable format
        binary_mask = result["binary_mask"]
        probs = result["probabilities"]

        # Compute statistics per class
        stats = {}
        for i, name in enumerate(result["class_names"]):
            if i < binary_mask.shape[0]:
                total_pixels = binary_mask[i].size
                positive_pixels = int(binary_mask[i].sum())
                stats[name] = {
                    "positive_pixels": positive_pixels,
                    "total_pixels": total_pixels,
                    "coverage_pct": round(
                        100.0 * positive_pixels / max(total_pixels, 1), 2
                    ),
                    "mean_confidence": round(
                        float(probs[i][binary_mask[i] > 0].mean())
                        if positive_pixels > 0
                        else 0.0,
                        4,
                    ),
                }

        # Build road graph & extract vector features
        from graph.road_graph import (
            build_road_graph_from_prediction,
            graph_to_geojson,
            extract_building_polygons,
        )

        G = build_road_graph_from_prediction(result)
        request.app.state.last_graph = G
        request.app.state.last_prediction = result

        road_geojson = graph_to_geojson(G)
        building_geojson = extract_building_polygons(
            binary_mask[0] if binary_mask.shape[0] > 0 else np.zeros((512, 512)),
            flooded_building_mask=binary_mask[3] if binary_mask.shape[0] > 3 else None,
        )

        # Convert masks and satellite photo to base64 PNG for frontend
        mask_b64 = _mask_to_base64_png(binary_mask)

        return JSONResponse(
            {
                "ok": True,
                "class_names": result["class_names"],
                "statistics": stats,
                "mask_png_base64": mask_b64,
                "mask_shape": list(binary_mask.shape),
                "road_network": road_geojson,
                "building_footprints": building_geojson,
                "num_road_segments": len(road_geojson.get("features", [])),
                "num_buildings": len(building_geojson.get("features", [])),
            }
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/route")
async def find_route(
    request: Request,
    start_y: int = Form(..., description="Start Y pixel coordinate"),
    start_x: int = Form(..., description="Start X pixel coordinate"),
    end_y: int = Form(..., description="End Y pixel coordinate"),
    end_x: int = Form(..., description="End X pixel coordinate"),
):
    """
    Find a safe route between two points avoiding flooded roads.

    Uses the most recent prediction's road/flood masks to build
    a road graph and find the shortest flood-free path.

    Returns:
        GeoJSON route with distance and flood status.
    """
    try:
        from inference.predict import create_stub_prediction
        from graph.road_graph import (
            build_road_graph_from_prediction,
            find_safe_route,
            route_to_geojson,
        )

        # Use stored prediction or live graph
        G = getattr(request.app.state, "last_graph", None)
        if G is None or G.number_of_nodes() == 0:
            prediction = getattr(request.app.state, "last_prediction", None) or create_stub_prediction(512, 512)
            G = build_road_graph_from_prediction(prediction)
            request.app.state.last_graph = G

        # Find safe route avoiding flooded roads
        route = find_safe_route(
            G,
            start=(start_y, start_x),
            end=(end_y, end_x),
            avoid_flooded=True,
        )

        # Convert to GeoJSON
        geojson = route_to_geojson(route)

        return JSONResponse(
            {
                "ok": True,
                "route": geojson,
                "distance_m": route.get("distance", 0),
                "has_flooded_segments": route.get("has_flooded_segments", False),
                "route_type": route.get("route_type", "unknown"),
                "path_nodes": route.get("path", []),
            }
        )

    except Exception as e:
        logger.error(f"Routing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sample")
async def sample_prediction(request: Request):
    """
    Return a sample prediction with realistic road vectors and flood zones.
    """
    from inference.predict import create_stub_prediction
    from graph.road_graph import (
        build_road_graph_from_prediction,
        graph_to_geojson,
        extract_building_polygons,
    )

    result = create_stub_prediction(512, 512)

    stats = {}
    for i, name in enumerate(result["class_names"]):
        if i < result["binary_mask"].shape[0]:
            positive = int(result["binary_mask"][i].sum())
            total = result["binary_mask"][i].size
            stats[name] = {
                "positive_pixels": positive,
                "coverage_pct": round(100.0 * positive / max(total, 1), 2),
            }

    mask_b64 = _mask_to_base64_png(result["binary_mask"])

    G = build_road_graph_from_prediction(result)
    request.app.state.last_graph = G
    request.app.state.last_prediction = result

    road_geojson = graph_to_geojson(G)
    building_geojson = extract_building_polygons(
        result["binary_mask"][0],
        flooded_building_mask=result["binary_mask"][3] if result["binary_mask"].shape[0] > 3 else None,
    )

    return JSONResponse(
        {
            "ok": True,
            "class_names": result["class_names"],
            "statistics": stats,
            "mask_png_base64": mask_b64,
            "road_network": road_geojson,
            "building_footprints": building_geojson,
            "num_road_segments": len(road_geojson.get("features", [])),
            "num_buildings": len(building_geojson.get("features", [])),
            "is_stub": True,
        }
    )


def _mask_to_base64_png(binary_mask: np.ndarray) -> str:
    """Convert a multi-class binary mask to base64-encoded PNG."""
    import base64

    try:
        from inference.eval import mask_to_rgba
        from PIL import Image

        rgba = mask_to_rgba(binary_mask, alpha=0.7)
        img = Image.fromarray(rgba, "RGBA")

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode("utf-8")

    except ImportError:
        # Fallback: return simple encoding
        import base64

        flat = binary_mask.astype(np.uint8).tobytes()
        return base64.b64encode(flat).decode("utf-8")
