"""
FastAPI Backend Application for SpaceNet 8 Flood Mapping.

Serves the prediction API and static frontend files.
Supports both real model inference and stub mode for development.
"""

import logging
import os
from pathlib import Path
import sys

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

from backend.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration from environment
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_DEBUG = os.getenv("API_DEBUG", "true").lower() == "true"
USE_STUB = os.getenv("USE_STUB", "true").lower() == "true"

# Application
app = FastAPI(
    title="SpaceNet 8 Flood Mapping API",
    description=(
        "AI-powered flood damage detection from satellite imagery. "
        "Upload pre/post event GeoTIFF pairs to get building footprints, "
        "road networks, and flood damage segmentation overlays."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

# Serve frontend static files (React build)
frontend_build = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_build.exists():
    app.mount("/", StaticFiles(directory=str(frontend_build), html=True))
    logger.info(f"Serving frontend from {frontend_build}")


@app.on_event("startup")
async def startup():
    """Initialize model on startup."""
    logger.info("=" * 60)
    logger.info("SpaceNet 8 Flood Mapping API Starting")
    logger.info(f"  Stub mode: {USE_STUB}")
    logger.info(f"  Debug: {API_DEBUG}")
    logger.info("=" * 60)

    if not USE_STUB:
        try:
            from inference.predict import FloodPredictor

            model_type = os.getenv("MODEL_TYPE", "fusion")
            env_key = "O2_BASELINE_WEIGHTS" if model_type == "unet" else "O3_FUSION_WEIGHTS"
            default = f"./weights/{'unet' if model_type == 'unet' else 'fusion'}_net_best.pth"

            # Resolve relative to the project root so the server can start from any cwd
            ckpt = Path(os.getenv(env_key, default))
            if not ckpt.is_absolute():
                ckpt = ROOT_DIR / ckpt
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"{env_key} checkpoint not found: {ckpt}. Download the weights into weights/."
                )

            app.state.predictor = FloodPredictor(
                checkpoint_path=ckpt,
                model_type=model_type,
            )
            logger.info(f"Loaded model: {model_type} from {ckpt}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Falling back to stub mode")
            app.state.predictor = None
    else:
        app.state.predictor = None
        logger.info("Running in STUB mode (no model loaded)")

    # Build the pre/post pair index so dataset tiles can be picked by name
    from backend.routes import get_pair_index

    get_pair_index(app)


def run():
    """Run the API server."""
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_DEBUG,
    )


if __name__ == "__main__":
    run()
