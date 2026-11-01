"""
Predictor for the research models trained by the experiment queue.

A run folder (weights/research/<run_id>/) holds the run's best.pt (from its
Kaggle output), config.yaml and metrics_val.json (from the results branch).
Inference follows the final evaluation exactly: the post image is resampled
onto the pre grid by georeferencing only, the tile is predicted by sliding
windows, and the run's validation-tuned thresholds and building/flood rules
turn probabilities into masks.
"""

import json
import logging
import time
from pathlib import Path
from typing import Union

import numpy as np
import torch
import yaml

from dataloader.align import load_aligned_pair
from training.inference import predict_tile, to_binary

logger = logging.getLogger(__name__)

RESEARCH_DIR = Path(__file__).resolve().parent.parent / "weights" / "research"
MODEL_LABELS = {
    "ra_fusion": "RA-Fusion (DINOv3 satellite ViT-L)",
    "sn8_baseline": "SpaceNet 8 baseline (ResNet34)",
    "siamese_unet": "Siamese U-Net (EfficientNet-B5)",
    "snunet": "SNUNet-CD",
    "unet_v0": "U-Net v0 (retrained)",
    "siamese_v0": "Siamese fusion v0 (retrained)",
}


def run_label(cfg: dict) -> str:
    return f"{MODEL_LABELS.get(cfg['model'], cfg['model'])} · {cfg['protocol']} seed {cfg['seed']}"


def list_research_runs(root: Path = None) -> list[dict]:
    """Run folders that have a config; 'available' once best.pt is in place."""
    root = root or RESEARCH_DIR
    runs = []
    for d in sorted(root.glob("*/")) if root.exists() else []:
        if not (d / "config.yaml").exists():
            continue
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        runs.append({
            "id": d.name,
            "name": run_label(cfg),
            "model": cfg["model"],
            "protocol": cfg["protocol"],
            "available": (d / "best.pt").exists(),
        })
    return runs


class ResearchPredictor:
    """
    Same interface as inference.predict.FloodPredictor (predict_from_files,
    model_type, checkpoint_name), for a trained research run.

    Args:
        run_dir: Folder with best.pt, config.yaml and metrics_val.json.
        device: 'auto', 'cuda' or 'cpu'.
        tta: Average over flips, as in the paper's test metrics (4x slower).
    """

    def __init__(self, run_dir: Union[str, Path], device: str = "auto", tta: bool = False):
        from models import build_research_model
        from training.engine import load_model_state

        run_dir = Path(run_dir)
        self.cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        self.thresholds = json.loads((run_dir / "metrics_val.json").read_text())["thresholds"]
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        self.tta = tta
        self.model_type = run_dir.name
        self.display_name = run_label(self.cfg)
        self.checkpoint_name = f"{run_dir.name}/best.pt"

        model = build_research_model(self.cfg["model"], **self.cfg.get("model_kwargs", {}))
        best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        load_model_state(model, best["model"])
        self.model = model.to(self.device).eval()
        logger.info(f"Research model {run_dir.name} ready on {self.device}")

    @torch.no_grad()
    def predict_from_files(self, pre_path: Union[str, Path], post_path: Union[str, Path]) -> dict:
        start = time.perf_counter()
        pair = load_aligned_pair(pre_path, post_path)
        pre, post = pair["pre"], pair["post"]
        ev = self.cfg["eval"]
        probs = predict_tile(
            self.model, pre, post, window=ev["window"], overlap=ev["overlap"], batch_size=ev["window_batch"],
            device=self.device, amp=self.cfg.get("amp", True), tta=self.tta,
        )
        binary = to_binary(probs, self.thresholds)
        flood = np.maximum(probs["flooded_building"], probs["flooded_road"])
        return {
            "probabilities": np.stack([probs["building"], probs["road"], flood,
                                       probs["flooded_building"], probs["flooded_road"]]),
            "binary_mask": binary,
            "class_names": ["building", "road", "flood", "flooded_building", "flooded_road"],
            "pre_rgb": pre,
            "post_rgb": post,
            "gsd_m": pair["pre_gsd_m"],
            "alignment": {"method": "georeferenced resampling (as in training and evaluation)", "applied": False},
            "flood_image_prob": round(float(probs["flood_image"]), 3),
            "inference_s": round(time.perf_counter() - start, 2),
        }
