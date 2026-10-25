"""
Prediction Pipeline for SpaceNet 8 Flood Mapping.

Handles end-to-end inference: load model → preprocess images →
sliding-window prediction → post-process → output masks.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from dataloader.align import apply_shift, estimate_shift_blocks, load_aligned_pair
from dataloader.dataset import read_geotiff, normalize_image, NUM_CLASSES

logger = logging.getLogger(__name__)

# Shifts beyond this are treated as registration failures (p90 on SN8 is ~40 px)
MAX_REGISTRATION_SHIFT_PX = 80.0

# Per-class thresholds calibrated during training/evaluation
# [building, road, flood, flooded_building, flooded_road]
DEFAULT_THRESHOLDS = [0.45, 0.38, 0.35, 0.30, 0.30]


def binarize(prob_map: np.ndarray, thresholds: list[float]) -> np.ndarray:
    """
    Apply per-class thresholds and the overlap heuristics used in evaluation.

    Flooded buildings/roads must lie on a building/road, and a building or
    road that overlaps predicted flood is attributed as flooded.

    Args:
        prob_map: Probabilities (NUM_CLASSES, H, W).
        thresholds: Per-class thresholds.

    Returns:
        Binary mask (NUM_CLASSES, H, W) uint8.
    """
    binary = np.zeros_like(prob_map, dtype=np.uint8)
    for c in range(min(prob_map.shape[0], len(thresholds))):
        binary[c] = prob_map[c] > thresholds[c]

    if binary.shape[0] >= 5:
        bldg, road, flood = binary[0], binary[1], binary[2]
        binary[3] = np.clip(binary[3] + bldg * flood, 0, 1) * bldg
        binary[4] = np.clip(binary[4] + road * flood, 0, 1) * road

    return binary


class FloodPredictor:
    """
    End-to-end flood prediction pipeline.

    Loads a trained model checkpoint, preprocesses input GeoTIFF pairs,
    runs sliding-window inference, and returns multi-class probability masks.

    Args:
        checkpoint_path: Path to the model checkpoint (.pth).
        model_type: Model architecture ('unet' or 'fusion').
        device: Torch device ('cuda' or 'cpu').
        tile_size: Sliding window tile size.
        overlap: Overlap between sliding window tiles.
        thresholds: Per-class binarization thresholds.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        model_type: str = "fusion",
        device: str = "auto",
        tile_size: int = 512,
        overlap: int = 64,
        thresholds: Optional[list[float]] = None,
    ):
        self.tile_size = tile_size
        self.overlap = overlap
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.model_type = model_type
        self.checkpoint_name = Path(checkpoint_path).name

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model
        self.model = self._load_model(checkpoint_path, model_type)
        self.model.eval()
        logger.info(
            f"FloodPredictor ready: {model_type} on {self.device}, "
            f"tile={tile_size}, overlap={overlap}"
        )

    def _load_model(
        self, checkpoint_path: Union[str, Path], model_type: str
    ) -> torch.nn.Module:
        """Load model from checkpoint."""
        from models import build_model

        if model_type in ("unet", "unet_baseline"):
            model = build_model("unet", in_channels=6, num_classes=NUM_CLASSES, pretrained=False)
        else:
            model = build_model("fusion", in_channels=3, num_classes=NUM_CLASSES, pretrained=False)

        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)

            # Clean DataParallel prefix; strict so an architecture mismatch fails loudly
            clean = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean, strict=True)
            logger.info(
                f"Loaded checkpoint: {checkpoint_path.name} "
                f"(epoch {ckpt.get('epoch', '?')}, best mIoU {ckpt.get('best_iou', '?')})"
            )
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}, using random weights")

        return model.to(self.device)

    @torch.no_grad()
    def predict(
        self,
        pre_image: np.ndarray,
        post_image: np.ndarray,
    ) -> dict:
        """
        Run prediction on a pre/post image pair.

        Args:
            pre_image: Pre-event image array (C, H, W) or (H, W, C), float32 [0, 1].
            post_image: Post-event image array, same format.

        Returns:
            Dictionary with:
              - 'probabilities': Probability map (NUM_CLASSES, H, W)
              - 'binary_mask': Binary predictions (NUM_CLASSES, H, W)
              - 'class_names': List of class names
        """
        # Ensure CHW format
        if pre_image.ndim == 3 and pre_image.shape[2] <= 4:
            pre_image = pre_image.transpose(2, 0, 1)
        if post_image.ndim == 3 and post_image.shape[2] <= 4:
            post_image = post_image.transpose(2, 0, 1)

        # Align spatial dimensions (use common overlapping area)
        _, h1, w1 = pre_image.shape
        _, h2, w2 = post_image.shape
        H = min(h1, h2)
        W = min(w1, w2)

        pre_image = pre_image[:, :H, :W]
        post_image = post_image[:, :H, :W]

        # Normalize
        pre_norm = normalize_image(pre_image.copy())
        post_norm = normalize_image(post_image.copy())

        # Sliding window inference
        prob_map = self._sliding_window_inference(pre_norm, post_norm, H, W)

        binary = binarize(prob_map, self.thresholds)

        return {
            "probabilities": prob_map,
            "binary_mask": binary,
            "class_names": ["building", "road", "flood", "flooded_building", "flooded_road"],
        }

    def predict_from_files(
        self,
        pre_path: Union[str, Path],
        post_path: Union[str, Path],
        register: bool = True,
    ) -> dict:
        """
        Run prediction on GeoTIFF file paths.

        The post-event image is resampled onto the pre-event pixel grid
        using the GeoTIFF georeferencing, then the residual global shift
        (median ~12 px on SpaceNet 8) is estimated by block-wise phase
        correlation and corrected when the estimate is confident.

        Args:
            pre_path: Path to pre-event GeoTIFF.
            post_path: Path to post-event GeoTIFF.
            register: Correct the residual shift after resampling.

        Returns:
            Prediction dictionary (same as predict()) plus 'pre_rgb' /
            'post_rgb' aligned uint8 previews and an 'alignment' report.
        """
        start = time.perf_counter()
        try:
            pair = load_aligned_pair(pre_path, post_path)
        except Exception as e:
            # Files without usable georeferencing: fall back to raw pixels
            logger.warning(f"Georeferenced alignment failed ({e}); using raw pixel grids")
            pre_img = read_geotiff(pre_path, max_bands=3)
            post_img = read_geotiff(post_path, max_bands=3)
            result = self.predict(pre_img, post_img)
            h, w = result["binary_mask"].shape[1:]
            result["pre_rgb"] = (pre_img[:, :h, :w] * 255).astype(np.uint8)
            result["post_rgb"] = (post_img[:, :h, :w] * 255).astype(np.uint8)
            result["alignment"] = {"method": "none (no georeferencing)", "applied": False}
            result["inference_s"] = round(time.perf_counter() - start, 2)
            return result

        post = pair["post"]
        alignment = {"method": "georeferenced resampling", "applied": False}
        if register:
            reg = estimate_shift_blocks(pair["pre"], post, pair["valid"])
            known = reg["n_confident"] >= 2 and reg["magnitude"] == reg["magnitude"]
            alignment.update({
                "method": "georeferenced resampling + block phase correlation",
                "residual_shift_px": round(reg["magnitude"], 1) if known else None,
                "dy_px": round(reg["dy"], 1) if known else None,
                "dx_px": round(reg["dx"], 1) if known else None,
                "confident_blocks": reg["n_confident"],
                "applied": bool(known and reg["magnitude"] <= MAX_REGISTRATION_SHIFT_PX),
            })
            if alignment["applied"]:
                post = apply_shift(post, reg["dy"], reg["dx"])

        result = self.predict(
            pair["pre"].astype(np.float32) / 255.0,
            post.astype(np.float32) / 255.0,
        )
        result["pre_rgb"] = pair["pre"]
        result["post_rgb"] = post
        result["alignment"] = alignment
        result["gsd_m"] = pair["pre_gsd_m"]
        result["inference_s"] = round(time.perf_counter() - start, 2)
        return result

    def _sliding_window_inference(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        H: int,
        W: int,
    ) -> np.ndarray:
        """Run sliding window inference with overlap averaging."""
        stride = self.tile_size - self.overlap
        result = np.zeros((NUM_CLASSES, H, W), dtype=np.float64)
        counts = np.zeros((H, W), dtype=np.float64)

        for y in range(0, H, stride):
            for x in range(0, W, stride):
                # Extract tile boundaries
                y_end = min(y + self.tile_size, H)
                x_end = min(x + self.tile_size, W)

                th = y_end - y
                tw = x_end - x

                # Always pad to exact tile_size x tile_size for consistent network input
                pre_tile = np.zeros((3, self.tile_size, self.tile_size), dtype=pre.dtype)
                post_tile = np.zeros((3, self.tile_size, self.tile_size), dtype=post.dtype)

                pre_tile[:, :th, :tw] = pre[:, y:y_end, x:x_end]
                post_tile[:, :th, :tw] = post[:, y:y_end, x:x_end]

                # Convert to tensor
                pre_t = torch.from_numpy(pre_tile).unsqueeze(0).to(self.device)
                post_t = torch.from_numpy(post_tile).unsqueeze(0).to(self.device)

                # Inference
                logits = self.model(pre_t, post_t)
                probs = torch.sigmoid(logits).cpu().numpy()[0]  # (C, tile_size, tile_size)

                # Accumulate only the valid unpadded region
                result[:, y:y_end, x:x_end] += probs[:, :th, :tw]
                counts[y:y_end, x:x_end] += 1.0

        # Average overlapping regions
        counts = np.maximum(counts, 1.0)
        result = result / counts[np.newaxis, :, :]

        return result.astype(np.float32)


def create_stub_prediction(height: int = 512, width: int = 512) -> dict:
    """
    Create a synthetic prediction for API testing without a trained model.

    Returns:
        Same format as FloodPredictor.predict().
    """
    rng = np.random.RandomState(42)

    # Generate realistic-looking patterns
    prob_map = np.zeros((NUM_CLASSES, height, width), dtype=np.float32)

    # Buildings: rectangular blobs
    for _ in range(20):
        y, x = rng.randint(0, height - 30), rng.randint(0, width - 30)
        h, w = rng.randint(10, 30), rng.randint(10, 30)
        prob_map[0, y : y + h, x : x + w] = rng.uniform(0.6, 0.95)

    # Roads: horizontal and vertical lines
    for _ in range(5):
        y = rng.randint(0, height)
        prob_map[1, max(0, y - 3) : y + 3, :] = rng.uniform(0.5, 0.9)
    for _ in range(5):
        x = rng.randint(0, width)
        prob_map[1, :, max(0, x - 3) : x + 3] = rng.uniform(0.5, 0.9)

    # Flood: large irregular region
    cy, cx = height // 2, width // 2
    Y, X = np.ogrid[:height, :width]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    prob_map[2] = np.clip(1.0 - dist / (height * 0.4), 0, 0.8)

    # Flooded buildings & roads
    prob_map[3] = prob_map[0] * prob_map[2]
    prob_map[4] = prob_map[1] * prob_map[2]

    binary = (prob_map > 0.5).astype(np.uint8)

    return {
        "probabilities": prob_map,
        "binary_mask": binary,
        "class_names": ["building", "road", "flood", "flooded_building", "flooded_road"],
    }
