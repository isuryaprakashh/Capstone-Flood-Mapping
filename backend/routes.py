"""
API Routes for SpaceNet 8 Flood Mapping.

Endpoints:
  GET  /api/health           — Health check and loaded model
  GET  /api/pairs/aois       — AOIs available in the dataset
  GET  /api/pairs            — Search pre/post pairs from the mapping CSVs
  GET  /api/pairs/lookup     — Pre image filename → its post image(s)
  GET  /api/pairs/image      — Download a dataset GeoTIFF from the index
  GET  /api/pairs/thumbnail  — JPEG preview of a dataset image
  POST /api/predict          — Upload pre/post GeoTIFFs → flood mapping
  POST /api/predict_pair     — Run on a dataset pair by name (no upload),
                               scored against ground truth when labelled
"""

import base64
import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response

logger = logging.getLogger(__name__)

router = APIRouter()

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_NAMES = {
    "fusion": "Siamese Cross-Attention Fusion (ResNet34)",
    "unet": "U-Net Baseline (ResNet34, early fusion)",
}


# ─── Pair Index ─────────────────────────────────────────────────────────────


def get_pair_index(app):
    """Return (and cache on app.state) the dataset's pre/post pair index."""
    index = getattr(app.state, "pair_index", None)
    if index is None:
        from dataloader.pairs import PairIndex

        data_root = Path(os.getenv("SPACENET8_DATA_DIR", ROOT_DIR / "CP1-DATASET"))
        if not data_root.is_absolute():
            data_root = ROOT_DIR / data_root
        index = PairIndex(
            data_root,
            alignment_stats=ROOT_DIR / "docs" / "analysis" / "alignment_stats.csv",
        )
        app.state.pair_index = index
    return index


def _pair_index(request: Request):
    return get_pair_index(request.app)


def _resolve_or_404(request: Request, aoi: str, kind: str, name: str) -> Path:
    if kind not in ("pre", "post"):
        raise HTTPException(status_code=400, detail="kind must be 'pre' or 'post'")
    path = _pair_index(request).resolve(aoi, kind, name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"{kind} image '{name}' not found in {aoi}")
    return path


# ─── Endpoints ──────────────────────────────────────────────────────────────


@router.get("/health")
async def health_check(request: Request):
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "SpaceNet 8 Flood Mapping API",
        "version": "1.1.0",
        "stub_mode": getattr(request.app.state, "predictor", None) is None,
        "model": _model_info(request),
    }


@router.get("/pairs/aois")
async def list_aois(request: Request):
    """List the AOIs (regions) found in the dataset mapping CSVs."""
    return {"ok": True, "aois": _pair_index(request).aois()}


@router.get("/pairs")
async def search_pairs(
    request: Request,
    aoi: Optional[str] = Query(None, description="AOI folder name"),
    q: Optional[str] = Query(None, description="Substring of pre image or tile name"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Search pre/post pairs; each pair lists its post images best-aligned first."""
    pairs, total = _pair_index(request).search(aoi=aoi, query=q, limit=limit, offset=offset)
    return {"ok": True, "total": total, "pairs": pairs}


@router.get("/pairs/lookup")
async def lookup_pair(request: Request, pre_image: str = Query(..., description="Pre image filename")):
    """Resolve a pre-event image filename to its mapped post-event image(s)."""
    entry = _pair_index(request).lookup(pre_image)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No mapping found for '{Path(pre_image).name}'")
    return {"ok": True, "pair": entry}


@router.get("/pairs/image")
async def pair_image(request: Request, aoi: str, kind: str, name: str):
    """Download a GeoTIFF that is listed in the dataset mapping CSVs."""
    path = _resolve_or_404(request, aoi, kind, name)
    return FileResponse(path, media_type="image/tiff", filename=path.name)


@router.get("/pairs/thumbnail")
async def pair_thumbnail(request: Request, aoi: str, kind: str, name: str):
    """Return a small JPEG preview of a dataset image."""
    path = _resolve_or_404(request, aoi, kind, name)
    b64 = await run_in_threadpool(_file_to_base64_rgb, path, 256)
    if not b64:
        raise HTTPException(status_code=500, detail="Could not render preview")
    return Response(content=base64.b64decode(b64), media_type="image/jpeg")


@router.post("/predict")
async def predict_flood(
    request: Request,
    pre_image: UploadFile = File(..., description="Pre-event GeoTIFF image"),
    post_image: UploadFile = File(..., description="Post-event GeoTIFF image"),
):
    """
    Run flood mapping on an uploaded pre/post GeoTIFF pair.

    Returns building footprints and the road network with flood status,
    an object-level damage summary, the raster overlay and image previews.
    """
    try:
        pre_content = await pre_image.read()
        post_content = await post_image.read()

        with tempfile.TemporaryDirectory() as tmpdir:
            pre_path = Path(tmpdir) / "pre.tif"
            post_path = Path(tmpdir) / "post.tif"
            pre_path.write_bytes(pre_content)
            post_path.write_bytes(post_content)

            result = await run_in_threadpool(_run_model, request, pre_path, post_path)
            payload = await run_in_threadpool(_prediction_payload, request, result)

        return JSONResponse(payload)

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict_pair")
async def predict_pair(
    request: Request,
    aoi: str = Form(..., description="AOI folder name"),
    pre_image: str = Form(..., description="Pre-event image filename"),
    post_image: Optional[str] = Form(None, description="Post image (default: best-aligned mapped post)"),
):
    """
    Run flood mapping on a dataset pair selected by name.

    Only the pre image is required: the post image is taken from the
    mapping CSV (best-aligned post image when the tile has two). For
    labelled AOIs the response also contains the ground truth and
    per-tile accuracy.
    """
    index = _pair_index(request)
    entry = index.lookup(pre_image)
    if entry is None or entry["aoi"] != aoi:
        raise HTTPException(status_code=404, detail=f"No mapping found for '{pre_image}' in {aoi}")

    post_name = post_image or entry["post_images"][0]
    pre_path = _resolve_or_404(request, aoi, "pre", pre_image)
    post_path = _resolve_or_404(request, aoi, "post", post_name)

    try:
        result = await run_in_threadpool(_run_model, request, pre_path, post_path)
        payload = await run_in_threadpool(_prediction_payload, request, result)
        payload["pair"] = {**entry, "post_image": post_name}

        annotations = index.annotation_paths(entry)
        if annotations is not None:
            payload["ground_truth"] = await run_in_threadpool(
                _ground_truth_payload, result, pre_path, *annotations
            )
        return JSONResponse(payload)

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Prediction Helpers ─────────────────────────────────────────────────────


def _model_info(request: Request) -> dict:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        return {"name": "Stub (no model loaded)", "checkpoint": None, "stub": True}
    return {
        "name": MODEL_NAMES.get(predictor.model_type, predictor.model_type),
        "checkpoint": predictor.checkpoint_name,
        "stub": False,
    }


def _run_model(request: Request, pre_path: Path, post_path: Path) -> dict:
    """Run the loaded model, or a stub prediction on the real aligned images."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is not None:
        return predictor.predict_from_files(pre_path, post_path)

    from dataloader.align import load_aligned_pair, read_rgb
    from inference.predict import create_stub_prediction

    try:
        pair = load_aligned_pair(pre_path, post_path)
        pre, post, gsd_m = pair["pre"], pair["post"], pair["pre_gsd_m"]
        alignment = {"method": "georeferenced resampling", "applied": False}
    except Exception:
        pre, _ = read_rgb(pre_path)
        post, _ = read_rgb(post_path)
        gsd_m, alignment = 0.5, {"method": "none (no georeferencing)", "applied": False}

    result = create_stub_prediction(*pre.shape[1:])
    result.update({"pre_rgb": pre, "post_rgb": post, "gsd_m": gsd_m, "alignment": alignment})
    return result


def _prediction_payload(request: Request, result: dict) -> dict:
    """
    JSON payload shared by /predict and /predict_pair: vector buildings and
    roads with flood status, object-level summary, overlay and previews.
    """
    from inference.postprocess import vectorize_prediction

    binary_mask = result["binary_mask"]
    gsd_m = float(result.get("gsd_m", 0.5))
    vectors = vectorize_prediction(result, gsd_m=gsd_m)

    return {
        "ok": True,
        "model": _model_info(request),
        "summary": {
            **vectors["summary"],
            "gsd_m": round(gsd_m, 3),
            "inference_s": result.get("inference_s"),
        },
        "building_footprints": vectors["buildings"],
        "road_network": vectors["roads"],
        "mask_png_base64": _mask_to_base64_png(binary_mask),
        "mask_shape": list(binary_mask.shape),
        "pre_image_base64": _array_to_base64_jpeg(result["pre_rgb"]) if "pre_rgb" in result else "",
        "post_image_base64": _array_to_base64_jpeg(result["post_rgb"]) if "post_rgb" in result else "",
        "alignment": result.get("alignment"),
        "ground_truth": None,
    }


def _ground_truth_payload(
    result: dict,
    pre_path: Path,
    geojson_path: Path,
    reference_csv: Optional[Path],
) -> dict:
    """Reference vectors, their summary and per-tile pixel accuracy."""
    from dataloader.ground_truth import load_ground_truth
    from inference.eval import compute_prediction_metrics
    from inference.postprocess import summarize

    gt = load_ground_truth(geojson_path, pre_path, reference_csv)
    metrics = None
    if gt["mask"].shape == result["binary_mask"].shape:
        m = compute_prediction_metrics(result, gt["mask"])
        # Object classes only (the 'flood' channel is the union of the flooded ones)
        metrics = {
            k: (None if v is None else round(v, 4))
            for k, v in m.items() if k.endswith(("_building", "_road"))
        }

    return {
        "building_footprints": gt["buildings"],
        "road_network": gt["roads"],
        "summary": summarize(gt["buildings"], gt["roads"]),
        "metrics": metrics,
    }


# ─── Image Encoding ─────────────────────────────────────────────────────────


def _array_to_base64_jpeg(rgb: np.ndarray, max_size: int = 1024) -> str:
    """Encode a (3, H, W) uint8 array as a base64 JPEG thumbnail."""
    from PIL import Image

    try:
        img = Image.fromarray(np.ascontiguousarray(np.transpose(rgb[:3], (1, 2, 0))))
        img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        logger.warning(f"Error encoding array preview: {e}")
        return ""


def _file_to_base64_rgb(file_path: Path, max_size: int = 768) -> str:
    """Read a GeoTIFF as RGB and encode a base64 JPEG thumbnail."""
    try:
        from dataloader.align import read_rgb

        rgb, _ = read_rgb(file_path)
        return _array_to_base64_jpeg(rgb, max_size)
    except Exception as e:
        logger.warning(f"Error encoding file {file_path}: {e}")
        return ""


def _mask_to_base64_png(binary_mask: np.ndarray) -> str:
    """Convert a multi-class binary mask to base64-encoded RGBA PNG."""
    from PIL import Image

    from inference.eval import mask_to_rgba

    rgba = mask_to_rgba(binary_mask, alpha=0.7)
    buffer = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
