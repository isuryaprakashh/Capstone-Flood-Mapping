"""
Full-Tile Inference for the Research Models.

Runs a model over a 1300 x 1300 tile with overlapping square windows (the
training crop size), averages per-pixel probabilities where windows overlap,
and converts probabilities into the binary masks consumed by
inference/postprocess.vectorize_prediction.
"""

import numpy as np
import torch

PIXEL_HEADS = ("building", "building_border", "road", "junction", "flooded_building", "flooded_road")

DEFAULT_THRESHOLDS = {
    "building": 0.5,
    "building_border": 0.5,
    "road": 0.5,
    "flooded_building": 0.5,
    "flooded_road": 0.5,
}


def window_starts(length: int, window: int, overlap: int) -> list[int]:
    """Window start offsets covering [0, length) with at least `overlap` px overlap."""
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, window - overlap))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


@torch.no_grad()
def predict_tile(
    model: torch.nn.Module,
    pre: np.ndarray,
    post: np.ndarray,
    window: int = 512,
    overlap: int = 64,
    batch_size: int = 4,
    device: torch.device = torch.device("cpu"),
    amp: bool = True,
    tta: bool = False,
) -> dict[str, np.ndarray]:
    """
    Predict per-pixel probabilities for a full tile.

    Args:
        model: Research model in eval mode.
        pre: Pre-event image (3, H, W) uint8.
        post: Post-event image (3, H, W) uint8 on the pre grid.
        window: Window size (the training crop size).
        overlap: Minimum overlap between windows.
        batch_size: Windows per forward pass.
        device: Inference device.
        amp: fp16 autocast on CUDA.
        tta: Average over the four flips (identity, horizontal, vertical, both).

    Returns:
        Dict head → (H, W) float32 probabilities for PIXEL_HEADS, plus
        'flood_image' → float (max over windows).
    """
    _, h, w = pre.shape
    boxes = [(y, x) for y in window_starts(h, window, overlap) for x in window_starts(w, window, overlap)]
    acc = {k: np.zeros((h, w), dtype=np.float32) for k in PIXEL_HEADS}
    count = np.zeros((h, w), dtype=np.float32)
    flood_image = 0.0

    to_tensor = lambda a: torch.from_numpy(a.astype(np.float32) / 255.0)
    use_amp = amp and device.type == "cuda"
    flips = [(), (3,), (2,), (2, 3)] if tta else [()]

    for i in range(0, len(boxes), batch_size):
        chunk = boxes[i:i + batch_size]
        pre_b = torch.stack([to_tensor(pre[:, y:y + window, x:x + window]) for y, x in chunk]).to(device)
        post_b = torch.stack([to_tensor(post[:, y:y + window, x:x + window]) for y, x in chunk]).to(device)
        probs = {k: 0.0 for k in PIXEL_HEADS}
        image_prob = 0.0
        for dims in flips:
            flip = (lambda t: torch.flip(t, dims)) if dims else (lambda t: t)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                out = model(flip(pre_b), flip(post_b))
            for k in PIXEL_HEADS:
                probs[k] = probs[k] + flip(torch.sigmoid(out[k].float()))
            image_prob = image_prob + torch.sigmoid(out["flood_image"].float())
        for k in PIXEL_HEADS:
            p = (probs[k] / len(flips)).cpu().numpy()[:, 0]
            for (y, x), tile_p in zip(chunk, p):
                acc[k][y:y + window, x:x + window] += tile_p
        for y, x in chunk:
            count[y:y + window, x:x + window] += 1
        flood_image = max(flood_image, float((image_prob / len(flips)).max()))

    result = {k: v / np.maximum(count, 1) for k, v in acc.items()}
    result["flood_image"] = flood_image
    return result


def to_binary(probs: dict, thresholds: dict | None = None) -> np.ndarray:
    """
    Binary masks in the channel order of inference/postprocess:
    [building, road, flood, flooded_building, flooded_road].

    Building instances are separated by removing predicted borders, and flood
    attribution is restricted to predicted objects.
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    building = (probs["building"] > t["building"]) & ~(probs["building_border"] > t["building_border"])
    road = probs["road"] > t["road"]
    flooded_building = (probs["flooded_building"] > t["flooded_building"]) & (probs["building"] > t["building"])
    flooded_road = (probs["flooded_road"] > t["flooded_road"]) & road
    flood = flooded_building | flooded_road
    return np.stack([building, road, flood, flooded_building, flooded_road]).astype(np.uint8)
