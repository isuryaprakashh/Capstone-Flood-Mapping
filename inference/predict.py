"""
Prediction Pipeline for SpaceNet 8 Flood Mapping.

Handles end-to-end inference: load model → preprocess images →
sliding-window prediction → post-process → output masks.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from data.dataset import read_geotiff, normalize_image, NUM_CLASSES

logger = logging.getLogger(__name__)


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
        threshold: Binarization threshold for output masks.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        model_type: str = "fusion",
        device: str = "auto",
        tile_size: int = 512,
        overlap: int = 64,
        threshold: float = 0.5,
    ):
        self.tile_size = tile_size
        self.overlap = overlap
        self.threshold = threshold

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

            # Clean DataParallel prefix
            clean = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean, strict=False)
            logger.info(f"Loaded checkpoint: {checkpoint_path.name}")
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
            post_image = post_image.transpose(2, 0, 1)

        _, H, W = pre_image.shape

        # Normalize
        pre_norm = normalize_image(pre_image.copy())
        post_norm = normalize_image(post_image.copy())

        # Sliding window inference
        prob_map = self._sliding_window_inference(pre_norm, post_norm, H, W)

        # Binary mask
        binary = (prob_map > self.threshold).astype(np.uint8)

        return {
            "probabilities": prob_map,
            "binary_mask": binary,
            "class_names": ["building", "road", "flood", "flooded_building", "flooded_road"],
        }

    def predict_from_files(
        self,
        pre_path: Union[str, Path],
        post_path: Union[str, Path],
    ) -> dict:
        """
        Run prediction on GeoTIFF file paths.

        Args:
            pre_path: Path to pre-event GeoTIFF.
            post_path: Path to post-event GeoTIFF.

        Returns:
            Prediction dictionary (same as predict()).
        """
        pre_img = read_geotiff(pre_path, max_bands=3)
        post_img = read_geotiff(post_path, max_bands=3)
        return self.predict(pre_img, post_img)

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
                # Extract tile
                y_end = min(y + self.tile_size, H)
                x_end = min(x + self.tile_size, W)

                pre_tile = pre[:, y:y_end, x:x_end]
                post_tile = post[:, y:y_end, x:x_end]

                # Pad if needed
                _, th, tw = pre_tile.shape
                if th < self.tile_size or tw < self.tile_size:
                    pre_padded = np.zeros(
                        (3, self.tile_size, self.tile_size), dtype=pre.dtype
                    )
                    post_padded = np.zeros_like(pre_padded)
                    pre_padded[:, :th, :tw] = pre_tile
                    post_padded[:, :th, :tw] = post_tile
                    pre_tile = pre_padded
                    post_tile = post_padded

                # Convert to tensor
                pre_t = torch.from_numpy(pre_tile).unsqueeze(0).to(self.device)
                post_t = torch.from_numpy(post_tile).unsqueeze(0).to(self.device)

                # Inference
                logits = self.model(pre_t, post_t)
                probs = torch.sigmoid(logits).cpu().numpy()[0]  # (C, TH, TW)

                # Accumulate
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
