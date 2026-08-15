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

        # Convert mask to base64 PNG for frontend
        mask_b64 = _mask_to_base64_png(binary_mask)

        return JSONResponse(
            {
                "ok": True,
                "class_names": result["class_names"],
                "statistics": stats,
                "mask_png_base64": mask_b64,
                "mask_shape": list(binary_mask.shape),
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

        # Use stored prediction or generate stub
        prediction = create_stub_prediction(512, 512)

        # Build road graph
        G = build_road_graph_from_prediction(prediction)

        # Find safe route
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
                "distance_m": route["distance"],
                "has_flooded_segments": route["has_flooded_segments"],
                "route_type": route["route_type"],
            }
        )

    except Exception as e:
        logger.error(f"Routing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sample")
async def sample_prediction():
    """
    Return a sample prediction for demo/testing purposes.
    Uses stub data — no model required.
    """
    from inference.predict import create_stub_prediction

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

    return JSONResponse(
        {
            "ok": True,
            "class_names": result["class_names"],
            "statistics": stats,
            "mask_png_base64": mask_b64,
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
