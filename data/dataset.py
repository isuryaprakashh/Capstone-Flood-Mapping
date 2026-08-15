"""
PyTorch Dataset and DataLoader for SpaceNet 8 Flood Mapping.

Provides a configurable dataset class that loads pre/post satellite image
pairs with their corresponding multi-channel segmentation masks.
Supports albumentations augmentations and geographic train/val/test splits.
"""

import csv
import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# Default normalization stats (ImageNet-like for satellite RGB)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NUM_CLASSES = 5  # building, road, flood, flooded_building, flooded_road


def read_geotiff(path: Union[str, Path], max_bands: int = 3) -> np.ndarray:
    """
    Read a GeoTIFF and return as a float32 numpy array.

    Args:
        path: Path to the GeoTIFF file.
        max_bands: Maximum number of bands to read.

    Returns:
        Array of shape (C, H, W) in float32, scaled to [0, 1].
    """
    with rasterio.open(path) as src:
        bands = min(src.count, max_bands)
        data = src.read(list(range(1, bands + 1))).astype(np.float32)

        # Handle different data ranges
        if data.max() > 1.0:
            # Satellite imagery often has values > 255 (e.g., 12-bit)
            # Clip to reasonable range and normalize
            data = np.clip(data, 0, 10000)
            data = data / 10000.0

    return data


def normalize_image(
    image: np.ndarray,
    mean: np.ndarray = MEAN,
    std: np.ndarray = STD,
) -> np.ndarray:
    """
    Normalize image with channel-wise mean and std.

    Args:
        image: Array of shape (C, H, W).
        mean: Per-channel mean.
        std: Per-channel std.

    Returns:
        Normalized array.
    """
    c = image.shape[0]
    for i in range(min(c, len(mean))):
        image[i] = (image[i] - mean[i]) / (std[i] + 1e-8)
    return image


def build_sample_manifest(
    dataset_dirs: list[Union[str, Path]],
    mask_dir: Optional[Union[str, Path]] = None,
) -> list[dict]:
    """
    Build a manifest of (pre_image, post_image, mask, annotation) samples
    from one or more SpaceNet 8 dataset directories.

    Args:
        dataset_dirs: List of dataset root directories.
        mask_dir: Optional directory containing precomputed masks.

    Returns:
        List of sample dictionaries.
    """
    samples = []

    for dataset_dir in dataset_dirs:
        dataset_dir = Path(dataset_dir)
        mapping_csvs = list(dataset_dir.glob("*_label_image_mapping.csv"))

        if not mapping_csvs:
            logger.warning(f"No mapping CSV found in {dataset_dir}")
            continue

        mapping_csv = mapping_csvs[0]
        pre_dir = dataset_dir / "PRE-event"
        post_dir = dataset_dir / "POST-event"
        annotations_dir = dataset_dir / "annotations"
        region = dataset_dir.name  # e.g., "Germany_Training_Public"

        with open(mapping_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get("label", "").strip()
                pre_img = row.get("pre-event image", "").strip()
                post_img = row.get("post-event image 1", "").strip()

                if not label or not pre_img or not post_img:
                    continue

                pre_path = pre_dir / pre_img
                post_path = post_dir / post_img
                geojson_path = annotations_dir / label

                if not pre_path.exists() or not post_path.exists():
                    continue

                sample = {
                    "pre_path": str(pre_path),
                    "post_path": str(post_path),
                    "region": region,
                    "tile_id": label.replace(".geojson", ""),
                }

                # Add geojson path if annotations exist
                if geojson_path.exists():
                    sample["geojson_path"] = str(geojson_path)

                # Add mask path if precomputed
                if mask_dir:
                    mask_stem = label.replace(".geojson", "")
                    mask_path = Path(mask_dir) / f"{mask_stem}_mask.tif"
                    if mask_path.exists():
                        sample["mask_path"] = str(mask_path)

                samples.append(sample)

    logger.info(f"Built manifest: {len(samples)} samples from {len(dataset_dirs)} dirs")
    return samples


def geographic_split(
    samples: list[dict],
    val_region: str = "Germany_Training_Public",
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split samples using geographic region holdout.

    Uses one region as validation, and randomly samples from the other
    for test. This tests domain shift robustness.

    Args:
        samples: Full sample manifest.
        val_region: Region name to use as validation set.
        test_fraction: Fraction of training data for test set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_samples, val_samples, test_samples).
    """
    rng = np.random.RandomState(seed)

    val_samples = [s for s in samples if s["region"] == val_region]
    remaining = [s for s in samples if s["region"] != val_region]

    # Shuffle and split remaining into train/test
    rng.shuffle(remaining)
    test_count = max(1, int(len(remaining) * test_fraction))

    test_samples = remaining[:test_count]
    train_samples = remaining[test_count:]

    logger.info(
        f"Geographic split: train={len(train_samples)}, "
        f"val={len(val_samples)} ({val_region}), "
        f"test={len(test_samples)}"
    )

    return train_samples, val_samples, test_samples


class SpaceNet8Dataset(Dataset):
    """
    PyTorch Dataset for SpaceNet 8 flood mapping.

    Loads pre-event and post-event satellite image pairs with their
    corresponding multi-channel segmentation masks.

    Args:
        samples: List of sample dictionaries from build_sample_manifest.
        transform: Optional albumentations transform pipeline.
        normalize: Whether to apply ImageNet normalization.
        tile_size: Expected tile size (for resize/padding).
        use_precomputed_masks: If True, loads masks from files.
        reference_csv_paths: Dict mapping region names to reference CSV paths.
    """

    def __init__(
        self,
        samples: list[dict],
        transform=None,
        normalize: bool = True,
        tile_size: int = 512,
        use_precomputed_masks: bool = True,
        reference_csv_paths: Optional[dict] = None,
    ):
        self.samples = samples
        self.transform = transform
        self.normalize = normalize
        self.tile_size = tile_size
        self.use_precomputed_masks = use_precomputed_masks
        self.reference_csv_paths = reference_csv_paths or {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Read pre-event and post-event images
        pre_img = read_geotiff(sample["pre_path"], max_bands=3)  # (3, H, W)
        post_img = read_geotiff(sample["post_path"], max_bands=3)

        # Ensure consistent spatial dimensions
        _, h, w = pre_img.shape
        _, ph, pw = post_img.shape

        # Use the smaller dimensions
        min_h = min(h, ph, self.tile_size)
        min_w = min(w, pw, self.tile_size)
        pre_img = pre_img[:, :min_h, :min_w]
        post_img = post_img[:, :min_h, :min_w]

        # Pad to tile_size if needed
        if min_h < self.tile_size or min_w < self.tile_size:
            pre_img = self._pad_to_size(pre_img)
            post_img = self._pad_to_size(post_img)

        # Load mask
        mask = self._load_mask(sample, pre_img.shape[1], pre_img.shape[2])

        # Apply augmentations (albumentations)
        if self.transform is not None:
            pre_img, post_img, mask = self._apply_transform(
                pre_img, post_img, mask
            )

        # Normalize images
        if self.normalize:
            pre_img = normalize_image(pre_img.copy())
            post_img = normalize_image(post_img.copy())

        return {
            "pre_image": torch.from_numpy(pre_img),
            "post_image": torch.from_numpy(post_img),
            "mask": torch.from_numpy(mask.astype(np.float32)),
            "tile_id": sample.get("tile_id", ""),
            "region": sample.get("region", ""),
        }

    def _pad_to_size(self, img: np.ndarray) -> np.ndarray:
        """Pad image to tile_size with zeros."""
        c, h, w = img.shape
        padded = np.zeros(
            (c, self.tile_size, self.tile_size), dtype=img.dtype
        )
        padded[:, :h, :w] = img
        return padded

    def _load_mask(
        self, sample: dict, height: int, width: int
    ) -> np.ndarray:
        """Load or generate mask for a sample."""
        # Try precomputed mask first
        if self.use_precomputed_masks and "mask_path" in sample:
            with rasterio.open(sample["mask_path"]) as src:
                mask = src.read().astype(np.float32)
                # Crop/pad to match image
                _, mh, mw = mask.shape
                if mh != height or mw != width:
                    result = np.zeros(
                        (mask.shape[0], height, width), dtype=np.float32
                    )
                    sh = min(mh, height)
                    sw = min(mw, width)
                    result[:, :sh, :sw] = mask[:, :sh, :sw]
                    mask = result

                # Pad channels if needed
                if mask.shape[0] < NUM_CLASSES:
                    padded = np.zeros(
                        (NUM_CLASSES, height, width), dtype=np.float32
                    )
                    padded[: mask.shape[0]] = mask
                    mask = padded

                return mask

        # Generate mask on-the-fly from GeoJSON
        if "geojson_path" in sample:
            from data.mask_generator import generate_mask

            ref_csv = self.reference_csv_paths.get(sample.get("region"))
            mask = generate_mask(
                geojson_path=sample["geojson_path"],
                reference_image_path=sample["pre_path"],
                reference_csv_path=ref_csv,
            )
            # Crop to match
            _, mh, mw = mask.shape
            result = np.zeros(
                (NUM_CLASSES, height, width), dtype=np.float32
            )
            sh = min(mh, height)
            sw = min(mw, width)
            result[:, :sh, :sw] = mask[:NUM_CLASSES, :sh, :sw]
            return result

        # No annotations — return empty mask
        return np.zeros((NUM_CLASSES, height, width), dtype=np.float32)

    def _apply_transform(
        self,
        pre_img: np.ndarray,
        post_img: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply albumentations transforms to both images and mask."""
        # albumentations expects (H, W, C) format
        pre_hwc = pre_img.transpose(1, 2, 0)
        post_hwc = post_img.transpose(1, 2, 0)
        mask_hwc = mask.transpose(1, 2, 0)

        # Use additional_targets to transform both images consistently
        result = self.transform(
            image=pre_hwc,
            image2=post_hwc,
            mask=mask_hwc,
        )

        pre_out = result["image"].transpose(2, 0, 1)
        post_out = result["image2"].transpose(2, 0, 1)
        mask_out = result["mask"].transpose(2, 0, 1)

        return pre_out, post_out, mask_out


def get_train_transforms(tile_size: int = 512):
    """
    Get albumentations training augmentation pipeline.

    Returns:
        Albumentations Compose transform.
    """
    try:
        import albumentations as A

        return A.Compose(
            [
                A.RandomCrop(
                    height=tile_size,
                    width=tile_size,
                    p=1.0 if tile_size < 1300 else 0.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.3,
                ),
                A.GaussNoise(p=0.2),
                A.GaussianBlur(blur_limit=(3, 5), p=0.1),
            ],
            additional_targets={"image2": "image"},
        )
    except ImportError:
        logger.warning("albumentations not installed, skipping augmentations")
        return None


def get_val_transforms(tile_size: int = 512):
    """Get validation transforms (center crop only)."""
    try:
        import albumentations as A

        return A.Compose(
            [
                A.CenterCrop(
                    height=tile_size,
                    width=tile_size,
                    p=1.0 if tile_size < 1300 else 0.0,
                ),
            ],
            additional_targets={"image2": "image"},
        )
    except ImportError:
        return None


def create_dataloaders(
    data_root: Union[str, Path],
    mask_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 8,
    tile_size: int = 512,
    num_workers: int = 4,
    val_region: str = "Germany_Training_Public",
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Args:
        data_root: Root directory containing dataset splits.
        mask_dir: Directory with precomputed masks.
        batch_size: Batch size for DataLoaders.
        tile_size: Tile/crop size.
        num_workers: Number of data loading workers.
        val_region: Region to use for validation holdout.
        pin_memory: Pin memory for GPU transfer.

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders.
    """
    data_root = Path(data_root)

    # Discover dataset directories
    dataset_dirs = []
    for child in sorted(data_root.iterdir()):
        if child.is_dir() and (child / "PRE-event").exists():
            # Only include training sets (with annotations)
            if (child / "annotations").exists():
                dataset_dirs.append(child)

    if not dataset_dirs:
        raise FileNotFoundError(
            f"No valid SpaceNet 8 dataset dirs found in {data_root}"
        )

    # Build manifest and split
    samples = build_sample_manifest(dataset_dirs, mask_dir)
    train_samples, val_samples, test_samples = geographic_split(
        samples, val_region=val_region
    )

    # Build reference CSV paths
    ref_csv_paths = {}
    for d in dataset_dirs:
        ref_csvs = list(d.glob("*_reference.csv"))
        if ref_csvs:
            ref_csv_paths[d.name] = str(ref_csvs[0])

    # Create datasets
    train_ds = SpaceNet8Dataset(
        samples=train_samples,
        transform=get_train_transforms(tile_size),
        normalize=True,
        tile_size=tile_size,
        reference_csv_paths=ref_csv_paths,
    )

    val_ds = SpaceNet8Dataset(
        samples=val_samples,
        transform=get_val_transforms(tile_size),
        normalize=True,
        tile_size=tile_size,
        reference_csv_paths=ref_csv_paths,
    )

    test_ds = SpaceNet8Dataset(
        samples=test_samples,
        transform=get_val_transforms(tile_size),
        normalize=True,
        tile_size=tile_size,
        reference_csv_paths=ref_csv_paths,
    )

    # Create DataLoaders
    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    logger.info(
        f"Created DataLoaders: train={len(train_ds)}, "
        f"val={len(val_ds)}, test={len(test_ds)}"
    )

    return loaders
