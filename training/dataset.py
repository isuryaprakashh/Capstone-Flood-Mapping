"""
Training and validation datasets over prepared SpaceNet 8 tiles.

Training samples are random crops of the 1300 x 1300 tiles with geometric
augmentation (shared by images and targets), independent photometric
augmentation of the pre and post acquisitions, synthetic clouds on the post
image, and synthetic misalignment: the post image is shifted by a random
offset so the model must learn to align. Each sample also carries a
self-supervised pair (post, post shifted by a known offset) used to train the
alignment module directly.
"""

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dataloader.prepare import TARGET_CHANNELS, load_tile

# Channels the loss consumes, in tensor order ('valid' is returned separately)
LABEL_CHANNELS = tuple(c for c in TARGET_CHANNELS if c != "valid")

DEFAULT_AUG = {
    "flip": True,
    "rot90": True,
    "photometric": True,
    "clouds": 0.15,
    "shift_prob": 0.5,
    "shift_max_px": 32,
    "object_crop_prob": 0.5,
    "flood_crop_prob": 0.5,  # share of object crops centred on flooded objects (when the tile has any)
}

DEFAULT_SSL = {"prob": 0.5, "max_px": 32}


def shift_image(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Translate an (H, W, C) or (H, W) image by (dy, dx) px, zero fill."""
    h, w = img.shape[:2]
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random brightness/contrast/saturation/gamma, noise and blur on (H, W, 3) float [0, 1]."""
    x = img * rng.uniform(0.8, 1.2) + rng.uniform(-0.08, 0.08)
    mean = x.mean()
    x = (x - mean) * rng.uniform(0.8, 1.2) + mean
    gray = x.mean(axis=2, keepdims=True)
    x = gray + (x - gray) * rng.uniform(0.8, 1.2)
    x = np.clip(x, 0, 1) ** rng.uniform(0.8, 1.25)
    if rng.random() < 0.3:
        x = x + rng.normal(0, rng.uniform(0.005, 0.03), x.shape)
    if rng.random() < 0.2:
        x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), rng.uniform(0.5, 1.5))
    return np.clip(x, 0, 1).astype(np.float32)


def add_clouds(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Blend smooth random haze patches toward white (thin clouds / fog)."""
    h, w = img.shape[:2]
    noise = rng.random((max(2, h // 64), max(2, w // 64))).astype(np.float32)
    field = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), max(h, w) / 32)
    field = (field - field.min()) / (field.max() - field.min() + 1e-6)
    alpha = np.clip((field - rng.uniform(0.4, 0.7)) * 3.0, 0, 1) * rng.uniform(0.5, 0.9)
    return (img * (1 - alpha[..., None]) + rng.uniform(0.85, 1.0) * alpha[..., None]).astype(np.float32)


class SN8CropDataset(Dataset):
    """
    Random training crops (indexed by tile; pair with a tile sampler).

    Args:
        processed_dir: Output folder of scripts/prepare_data.py.
        tile_ids: Tiles to sample from.
        crop_size: Square crop size (multiple of 32).
        aug: Augmentation settings (see DEFAULT_AUG).
        ssl: Self-supervised alignment pair settings (see DEFAULT_SSL);
            None disables the pair.
        seed: Base seed; combined with the epoch via set_epoch().
        register_post: Apply the stored classical registration to post images.
    """

    def __init__(
        self,
        processed_dir: Union[str, Path],
        tile_ids: list[str],
        crop_size: int = 512,
        aug: Optional[dict] = None,
        ssl: Optional[dict] = None,
        seed: int = 0,
        register_post: bool = False,
    ):
        self.processed_dir = Path(processed_dir)
        self.register_post = register_post
        self.tile_ids = list(tile_ids)
        self.crop_size = crop_size
        self.aug = {**DEFAULT_AUG, **(aug or {})}
        self.ssl = None if ssl is None else {**DEFAULT_SSL, **ssl}
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.tile_ids)

    def _rng(self, index: int) -> np.random.Generator:
        worker = torch.utils.data.get_worker_info()
        stream = worker.seed % (2 ** 32) if worker is not None else 0
        return np.random.default_rng([self.seed, self.epoch, index, stream])

    def _crop_origin(self, targets: dict, rng: np.random.Generator) -> tuple[int, int]:
        h, w = targets["valid"].shape
        c = self.crop_size
        for _ in range(4):
            if rng.random() < self.aug["object_crop_prob"]:
                objects = np.flatnonzero((targets["flooded_building"] | targets["flooded_road"]).ravel()[::97])
                if not objects.size or rng.random() >= self.aug["flood_crop_prob"]:
                    objects = np.flatnonzero((targets["building"] | targets["road"]).ravel()[::97])
                if objects.size:
                    cy, cx = divmod(int(objects[rng.integers(objects.size)]) * 97, w)
                    y = int(np.clip(cy - c // 2 + rng.integers(-c // 4, c // 4 + 1), 0, h - c))
                    x = int(np.clip(cx - c // 2 + rng.integers(-c // 4, c // 4 + 1), 0, w - c))
                else:
                    y, x = int(rng.integers(0, h - c + 1)), int(rng.integers(0, w - c + 1))
            else:
                y, x = int(rng.integers(0, h - c + 1)), int(rng.integers(0, w - c + 1))
            if targets["valid"][y:y + c, x:x + c].mean() >= 0.5:
                break
        return y, x

    def __getitem__(self, index: int) -> dict:
        # index = draw * len(self) + tile: repeated draws of a tile get different crops
        rng = self._rng(index)
        tile = load_tile(self.processed_dir, self.tile_ids[index % len(self.tile_ids)], self.register_post)
        y, x = self._crop_origin(tile["targets"], rng)
        c = self.crop_size

        pre = tile["pre"][:, y:y + c, x:x + c].transpose(1, 2, 0).astype(np.float32) / 255.0
        post = tile["post"][:, y:y + c, x:x + c].transpose(1, 2, 0).astype(np.float32) / 255.0
        labels = np.stack([tile["targets"][k][y:y + c, x:x + c] for k in LABEL_CHANNELS], axis=-1).astype(np.float32)
        valid = tile["targets"]["valid"][y:y + c, x:x + c].astype(np.float32)

        # Geometric augmentation, identical for every array
        if self.aug["flip"]:
            if rng.random() < 0.5:
                pre, post, labels, valid = (a[:, ::-1] for a in (pre, post, labels, valid))
            if rng.random() < 0.5:
                pre, post, labels, valid = (a[::-1] for a in (pre, post, labels, valid))
        if self.aug["rot90"]:
            k = int(rng.integers(4))
            pre, post, labels, valid = (np.rot90(a, k) for a in (pre, post, labels, valid))
        pre, post, labels, valid = (np.ascontiguousarray(a) for a in (pre, post, labels, valid))

        # Independent photometric augmentation: different acquisitions
        if self.aug["photometric"]:
            pre = photometric(pre, rng)
            post = photometric(post, rng)
        if rng.random() < self.aug["clouds"]:
            post = add_clouds(post, rng)

        # Synthetic misalignment: labels stay on the pre grid
        if rng.random() < self.aug["shift_prob"]:
            m = self.aug["shift_max_px"]
            dy, dx = rng.uniform(-m, m, size=2)
            post = shift_image(post, dy, dx)
            valid = valid * shift_image(np.ones_like(valid), dy, dx).round()

        sample = {
            "pre": torch.from_numpy(pre.transpose(2, 0, 1).copy()),
            "post": torch.from_numpy(post.transpose(2, 0, 1).copy()),
            "labels": torch.from_numpy(labels.transpose(2, 0, 1).copy()),
            "valid": torch.from_numpy(valid[None].copy()),
            "flood_image": torch.tensor([float(labels[..., LABEL_CHANNELS.index("flooded_building")].any()
                                               or labels[..., LABEL_CHANNELS.index("flooded_road")].any())]),
        }

        if self.ssl is not None:
            use = rng.random() < self.ssl["prob"]
            m = self.ssl["max_px"]
            dy, dx = rng.uniform(-m, m, size=2) if use else (0.0, 0.0)
            shifted = shift_image(post, dy, dx)
            if use and self.aug["photometric"]:
                shifted = photometric(shifted, rng)
            sample["ssl_a"] = sample["post"]
            sample["ssl_b"] = torch.from_numpy(shifted.transpose(2, 0, 1).copy())
            sample["ssl_flow"] = torch.tensor([dx, dy], dtype=torch.float32)
            sample["ssl"] = torch.tensor(float(use))
        return sample


class SN8CenterCropDataset(Dataset):
    """Deterministic centre crops for fast per-epoch validation."""

    def __init__(
        self,
        processed_dir: Union[str, Path],
        tile_ids: list[str],
        crop_size: int = 512,
        register_post: bool = False,
    ):
        self.processed_dir = Path(processed_dir)
        self.tile_ids = list(tile_ids)
        self.crop_size = crop_size
        self.register_post = register_post

    def __len__(self) -> int:
        return len(self.tile_ids)

    def __getitem__(self, index: int) -> dict:
        tile = load_tile(self.processed_dir, self.tile_ids[index], self.register_post)
        h, w = tile["targets"]["valid"].shape
        c = self.crop_size
        y, x = (h - c) // 2, (w - c) // 2
        labels = np.stack([tile["targets"][k][y:y + c, x:x + c] for k in LABEL_CHANNELS]).astype(np.float32)
        return {
            "pre": torch.from_numpy(tile["pre"][:, y:y + c, x:x + c].astype(np.float32) / 255.0),
            "post": torch.from_numpy(tile["post"][:, y:y + c, x:x + c].astype(np.float32) / 255.0),
            "labels": torch.from_numpy(labels),
            "valid": torch.from_numpy(tile["targets"]["valid"][None, y:y + c, x:x + c].astype(np.float32)),
            "flood_image": torch.tensor([float(labels[LABEL_CHANNELS.index("flooded_building")].any()
                                               or labels[LABEL_CHANNELS.index("flooded_road")].any())]),
        }


def tile_weights(manifest_tiles: dict[str, dict], tile_ids: list[str], flood_oversample: float = 2.0) -> torch.Tensor:
    """Sampling weights: tiles containing flooding are drawn `flood_oversample` times as often."""
    return torch.tensor(
        [flood_oversample if manifest_tiles[t].get("has_flood") else 1.0 for t in tile_ids],
        dtype=torch.double,
    )
