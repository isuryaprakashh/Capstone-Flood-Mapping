"""
API Routes for SpaceNet 8 Flood Mapping.

Endpoints:
  GET  /api/health           — Health check and loaded model
  GET  /api/models           — Models that can be selected for prediction
  GET  /api/pairs/aois       — AOIs available in the dataset
  GET  /api/pairs            — Search pre/post pairs from the mapping CSVs
  GET  /api/pairs/lookup     — Pre image filename → its post image(s)
  GET  /api/pairs/image      — Download a dataset GeoTIFF from the index
  GET  /api/pairs/thumbnail  — JPEG preview of a dataset image
  POST /api/predict          — Upload pre/post GeoTIFFs → flood mapping
  POST /api/predict_pair     — Run on a dataset pair by name (no upload),
                               scored against ground truth when labelled

Both prediction routes take an optional 'model' form field (an id from
/api/models); without it the model loaded at startup is used.
"""

import base64
import io
import logging
import os
import tempfile
import threading
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
LEGACY_WEIGHTS = {"fusion": ("O3_FUSION_WEIGHTS", "weights/fusion_net_best.pth"),
                  "unet": ("O2_BASELINE_WEIGHTS", "weights/unet_net_best.pth")}
_load_lock = threading.Lock()  # models load once, even when two requests ask at the same time


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
        "model": _model_info(getattr(request.app.state, "predictor", None)),
    }


@router.get("/models")
async def list_models(request: Request):
    """
    Selectable models: the two original models (weights/*.pth) and the
    research runs set up under weights/research/. A research run is
    'available' once its best.pt has been downloaded from Kaggle.
    """
    from inference.research import list_research_runs

    default = getattr(request.app.state, "predictor", None)
    models = []
    for model_id, (env_key, fallback) in LEGACY_WEIGHTS.items():
        models.append({"id": model_id, "name": f"{MODEL_NAMES[model_id]} · v0", "group": "Original models",
                       "available": _legacy_checkpoint(env_key, fallback).exists()})
    for run in list_research_runs():
        models.append({"id": run["id"], "name": run["name"], "group": "Research models", "available": run["available"]})
    return {"models": models, "default": default.model_type if default else None}

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
    model: Optional[str] = Form(None, description="Model id from /api/models (default: startup model)"),
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

            result = await run_in_threadpool(_run_model, request, pre_path, post_path, model)
            payload = await run_in_threadpool(_prediction_payload, request, result)

        return JSONResponse(payload)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict_pair")
async def predict_pair(
    request: Request,
    aoi: str = Form(..., description="AOI folder name"),
    pre_image: str = Form(..., description="Pre-event image filename"),
    post_image: Optional[str] = Form(None, description="Post image (default: best-aligned mapped post)"),
    model: Optional[str] = Form(None, description="Model id from /api/models (default: startup model)"),
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
        result = await run_in_threadpool(_run_model, request, pre_path, post_path, model)
        payload = await run_in_threadpool(_prediction_payload, request, result)
        payload["pair"] = {**entry, "post_image": post_name}

        annotations = index.annotation_paths(entry)
        if annotations is not None:
            payload["ground_truth"] = await run_in_threadpool(
                _ground_truth_payload, result, pre_path, *annotations
            )
        return JSONResponse(payload)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Prediction Helpers ─────────────────────────────────────────────────────


def _model_info(predictor) -> dict:
    if predictor is None:
        return {"name": "Stub (no model loaded)", "checkpoint": None, "stub": True}
    name = getattr(predictor, "display_name", None) or MODEL_NAMES.get(predictor.model_type, predictor.model_type)
    return {"id": predictor.model_type, "name": name, "checkpoint": predictor.checkpoint_name, "stub": False}


def _legacy_checkpoint(env_key: str, fallback: str) -> Path:
    path = Path(os.getenv(env_key, fallback))
    return path if path.is_absolute() else ROOT_DIR / path


def _predictor(request: Request, model_id: Optional[str]):
    """The predictor for a model id (loaded on first use and cached); None = stub."""
    state = request.app.state
    default = getattr(state, "predictor", None)
    if not model_id or (default is not None and model_id == default.model_type):
        return default
    with _load_lock:
        cache = getattr(state, "predictors", None)
        if cache is None:
            cache = state.predictors = {}
        if model_id not in cache:
            if model_id in LEGACY_WEIGHTS:
                from inference.predict import FloodPredictor

                ckpt = _legacy_checkpoint(*LEGACY_WEIGHTS[model_id])
                if not ckpt.exists():
                    raise HTTPException(status_code=404, detail=f"Weights for '{model_id}' not found: {ckpt}")
                cache[model_id] = FloodPredictor(checkpoint_path=ckpt, model_type=model_id)
            else:
                from inference.research import RESEARCH_DIR, ResearchPredictor

                run_dir = RESEARCH_DIR / model_id
                if model_id != run_dir.name or not (run_dir / "config.yaml").exists():
                    raise HTTPException(status_code=404, detail=f"Unknown model '{model_id}'")
                if not (run_dir / "best.pt").exists():
                    raise HTTPException(status_code=404, detail=f"Download best.pt into {run_dir} to use '{model_id}'")
                cache[model_id] = ResearchPredictor(run_dir)
        return cache[model_id]


def _run_model(request: Request, pre_path: Path, post_path: Path, model_id: Optional[str] = None) -> dict:
    """Run the selected (or startup) model, or a stub prediction on the real aligned images."""
    predictor = _predictor(request, model_id)
    if predictor is not None:
        result = predictor.predict_from_files(pre_path, post_path)
        result["model_info"] = _model_info(predictor)
        return result

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
        "model": result.get("model_info") or _model_info(None),
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
