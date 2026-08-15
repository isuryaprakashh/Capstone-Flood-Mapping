"""
Data module for SpaceNet 8 Flood Mapping Pipeline.

Provides CRS reprojection, image tiling, mask generation,
and PyTorch Dataset/DataLoader utilities.
"""

from data.crs import (
    get_crs_info,
    reproject_raster,
    check_crs_alignment,
    batch_reproject,
)
from data.tiling import (
    tile_geotiff,
    tile_image_and_mask,
    reconstruct_from_tiles,
    compute_tile_windows,
)
from data.mask_generator import (
    generate_mask,
    batch_generate_masks,
    parse_geojson_annotations,
    parse_reference_csv,
    MASK_CHANNELS,
    NUM_MASK_CHANNELS,
)
from data.dataset import (
    SpaceNet8Dataset,
    create_dataloaders,
    build_sample_manifest,
    geographic_split,
    get_train_transforms,
    get_val_transforms,
    NUM_CLASSES,
)

__all__ = [
    # CRS
    "get_crs_info",
    "reproject_raster",
    "check_crs_alignment",
    "batch_reproject",
    # Tiling
    "tile_geotiff",
    "tile_image_and_mask",
    "reconstruct_from_tiles",
    "compute_tile_windows",
    # Mask Generation
    "generate_mask",
    "batch_generate_masks",
    "parse_geojson_annotations",
    "parse_reference_csv",
    "MASK_CHANNELS",
    "NUM_MASK_CHANNELS",
    # Dataset
    "SpaceNet8Dataset",
    "create_dataloaders",
    "build_sample_manifest",
    "geographic_split",
    "get_train_transforms",
    "get_val_transforms",
    "NUM_CLASSES",
]
