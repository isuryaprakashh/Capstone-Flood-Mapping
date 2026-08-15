"""
Mask Generator for SpaceNet 8 Annotations.

Converts GeoJSON annotation files and reference CSVs into multi-channel
binary segmentation masks for training. Generates 5 mask channels:
  - Channel 0: Building footprints
  - Channel 1: Road network
  - Channel 2: Flood extent (overall)
  - Channel 3: Flooded buildings
  - Channel 4: Flooded roads
"""

import csv
import json
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from shapely.geometry import LineString, Polygon, shape, mapping
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

# Mask channel definitions
MASK_CHANNELS = {
    "building": 0,
    "road": 1,
    "flood": 2,
    "flooded_building": 3,
    "flooded_road": 4,
}
NUM_MASK_CHANNELS = len(MASK_CHANNELS)

# Road line width in pixels for rasterization
ROAD_LINE_WIDTH = 8


def parse_geojson_annotations(
    geojson_path: Union[str, Path],
) -> dict[str, list]:
    """
    Parse a GeoJSON annotation file into categorized feature lists.

    Args:
        geojson_path: Path to the GeoJSON annotation file.

    Returns:
        Dictionary with keys: buildings, roads, flooded_buildings, flooded_roads.
    """
    geojson_path = Path(geojson_path)

    with open(geojson_path, "r") as f:
        data = json.load(f)

    result = {
        "buildings": [],
        "roads": [],
        "flooded_buildings": [],
        "flooded_roads": [],
    }

    for feature in data.get("features", []):
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})

        if not geom or not geom.get("coordinates"):
            continue

        geom_type = geom.get("type", "")
        is_flooded = str(props.get("flooded", "")).lower() in (
            "true",
            "yes",
            "1",
        )
        is_building = props.get("building") is not None
        is_road = props.get("highway") is not None

        try:
            shp = shape(geom)
            if shp.is_empty:
                continue
        except Exception:
            continue

        if is_building and geom_type == "Polygon":
            result["buildings"].append(shp)
            if is_flooded:
                result["flooded_buildings"].append(shp)
        elif is_road and geom_type == "LineString":
            result["roads"].append(shp)
            if is_flooded:
                result["flooded_roads"].append(shp)

    return result


def parse_reference_csv(
    csv_path: Union[str, Path],
    image_id: str,
) -> dict[str, list]:
    """
    Parse the SpaceNet 8 reference CSV to extract road geometries with
    flood status for a specific image.

    The reference CSV has columns:
      ImageId, Object, Wkt_Pix, Flooded, length_m, travel_time_s

    Args:
        csv_path: Path to the reference CSV.
        image_id: The ImageId to filter for.

    Returns:
        Dictionary with roads and flooded_roads in pixel coordinates.
    """
    from shapely import wkt

    result = {"roads_pix": [], "flooded_roads_pix": []}

    csv_path = Path(csv_path)
    if not csv_path.exists():
        return result

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("ImageId", "").strip() != image_id:
                continue

            obj_type = row.get("Object", "").strip()
            wkt_pix = row.get("Wkt_Pix", "").strip()
            flooded = row.get("Flooded", "").strip()

            if obj_type != "Road" or not wkt_pix or wkt_pix == "LINESTRING EMPTY":
                continue

            try:
                geom = wkt.loads(wkt_pix)
                if geom.is_empty:
                    continue
            except Exception:
                continue

            result["roads_pix"].append(geom)
            if flooded.lower() in ("true", "yes", "1"):
                result["flooded_roads_pix"].append(geom)

    return result


def rasterize_geometries(
    geometries: list,
    transform: rasterio.transform.Affine,
    width: int,
    height: int,
    burn_value: int = 1,
    buffer_size: Optional[float] = None,
) -> np.ndarray:
    """
    Rasterize a list of Shapely geometries onto a numpy array.

    Args:
        geometries: List of Shapely geometry objects.
        transform: Affine transform for the output raster.
        width: Output raster width.
        height: Output raster height.
        burn_value: Value to burn into the raster.
        buffer_size: Optional buffer to apply to geometries.

    Returns:
        Binary numpy array (height, width) with rasterized geometries.
    """
    if not geometries:
        return np.zeros((height, width), dtype=np.uint8)

    if buffer_size is not None:
        geometries = [g.buffer(buffer_size) for g in geometries]

    # Filter out empty geometries
    shapes = [(mapping(g), burn_value) for g in geometries if not g.is_empty]

    if not shapes:
        return np.zeros((height, width), dtype=np.uint8)

    mask = features.rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )

    return mask


def rasterize_pixel_geometries(
    geometries: list,
    width: int,
    height: int,
    burn_value: int = 1,
    buffer_size: float = 4.0,
) -> np.ndarray:
    """
    Rasterize geometries that are already in pixel coordinates.

    Args:
        geometries: List of Shapely geometries in pixel coords.
        width: Output width.
        height: Output height.
        burn_value: Burn value.
        buffer_size: Buffer in pixels (for LineStrings → thick lines).

    Returns:
        Binary mask array.
    """
    if not geometries:
        return np.zeros((height, width), dtype=np.uint8)

    # Use identity transform for pixel-coordinate geometries
    from rasterio.transform import Affine

    identity = Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    # Buffer linestrings to create road width
    buffered = []
    for g in geometries:
        if g.is_empty:
            continue
        if g.geom_type == "LineString":
            buffered.append(g.buffer(buffer_size))
        else:
            buffered.append(g)

    shapes = [(mapping(g), burn_value) for g in buffered if not g.is_empty]
    if not shapes:
        return np.zeros((height, width), dtype=np.uint8)

    mask = features.rasterize(
        shapes,
        out_shape=(height, width),
        transform=identity,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )

    return mask


def generate_mask(
    geojson_path: Union[str, Path],
    reference_image_path: Union[str, Path],
    reference_csv_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    road_buffer_pixels: float = 4.0,
) -> np.ndarray:
    """
    Generate a multi-channel binary segmentation mask from annotations.

    Produces a 5-channel mask:
      [0] Building footprints
      [1] Road network
      [2] Flood extent (union of flooded buildings + roads)
      [3] Flooded buildings
      [4] Flooded roads

    Args:
        geojson_path: Path to the GeoJSON annotation file.
        reference_image_path: Path to the reference GeoTIFF (for dimensions/transform).
        reference_csv_path: Optional path to the reference CSV for pixel-space roads.
        output_path: Optional path to save the mask as a GeoTIFF.
        road_buffer_pixels: Buffer width for road rasterization.

    Returns:
        Numpy array of shape (NUM_MASK_CHANNELS, H, W) with binary values.
    """
    geojson_path = Path(geojson_path)
    reference_image_path = Path(reference_image_path)

    # Get reference image metadata
    with rasterio.open(reference_image_path) as src:
        width = src.width
        height = src.height
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()

    # Parse GeoJSON annotations (geographic coordinates)
    annotations = parse_geojson_annotations(geojson_path)

    # Initialize mask channels
    mask = np.zeros((NUM_MASK_CHANNELS, height, width), dtype=np.uint8)

    # Channel 0: Buildings (from GeoJSON polygons)
    mask[MASK_CHANNELS["building"]] = rasterize_geometries(
        annotations["buildings"], transform, width, height
    )

    # Channel 3: Flooded buildings
    mask[MASK_CHANNELS["flooded_building"]] = rasterize_geometries(
        annotations["flooded_buildings"], transform, width, height
    )

    # Roads: prefer pixel-space reference CSV if available
    if reference_csv_path and Path(reference_csv_path).exists():
        # Extract image ID from the reference image filename
        image_id = reference_image_path.stem
        ref_data = parse_reference_csv(reference_csv_path, image_id)

        # Channel 1: Roads from pixel-space WKT
        mask[MASK_CHANNELS["road"]] = rasterize_pixel_geometries(
            ref_data["roads_pix"],
            width,
            height,
            buffer_size=road_buffer_pixels,
        )

        # Channel 4: Flooded roads from pixel-space WKT
        mask[MASK_CHANNELS["flooded_road"]] = rasterize_pixel_geometries(
            ref_data["flooded_roads_pix"],
            width,
            height,
            buffer_size=road_buffer_pixels,
        )
    else:
        # Fallback: use GeoJSON roads (geographic coords)
        mask[MASK_CHANNELS["road"]] = rasterize_geometries(
            annotations["roads"],
            transform,
            width,
            height,
            buffer_size=0.00003,  # ~3m buffer in geographic coords
        )

        mask[MASK_CHANNELS["flooded_road"]] = rasterize_geometries(
            annotations["flooded_roads"],
            transform,
            width,
            height,
            buffer_size=0.00003,
        )

    # Channel 2: Flood extent (union of flooded buildings + flooded roads)
    mask[MASK_CHANNELS["flood"]] = np.clip(
        mask[MASK_CHANNELS["flooded_building"]]
        + mask[MASK_CHANNELS["flooded_road"]],
        0,
        1,
    )

    # Save if output path specified
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        profile.update(
            count=NUM_MASK_CHANNELS,
            dtype="uint8",
            compress="lzw",
            nodata=0,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mask)
            # Write channel descriptions
            dst.update_tags(
                ns="mask",
                ch0="building",
                ch1="road",
                ch2="flood",
                ch3="flooded_building",
                ch4="flooded_road",
            )

        logger.info(f"Generated mask: {output_path.name} ({mask.sum()} positive pixels)")

    return mask


def batch_generate_masks(
    dataset_dir: Union[str, Path],
    output_dir: Union[str, Path],
    road_buffer_pixels: float = 4.0,
) -> list[Path]:
    """
    Generate masks for an entire SpaceNet 8 dataset directory.

    Expected dataset structure:
        dataset_dir/
        ├── PRE-event/         (GeoTIFF imagery)
        ├── POST-event/        (GeoTIFF imagery)
        ├── annotations/       (GeoJSON labels)
        ├── *_label_image_mapping.csv
        └── *_reference.csv

    Args:
        dataset_dir: Root directory of the dataset split.
        output_dir: Output directory for generated masks.
        road_buffer_pixels: Buffer width for road rasterization.

    Returns:
        List of paths to generated mask files.
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find the mapping CSV
    mapping_csvs = list(dataset_dir.glob("*_label_image_mapping.csv"))
    if not mapping_csvs:
        raise FileNotFoundError(
            f"No label_image_mapping.csv found in {dataset_dir}"
        )
    mapping_csv = mapping_csvs[0]

    # Find the reference CSV (optional)
    ref_csvs = list(dataset_dir.glob("*_reference.csv"))
    ref_csv = ref_csvs[0] if ref_csvs else None

    annotations_dir = dataset_dir / "annotations"
    pre_dir = dataset_dir / "PRE-event"

    generated = []
    skipped = 0

    with open(mapping_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_name = row.get("label", "").strip()
            pre_image_name = row.get("pre-event image", "").strip()

            if not label_name or not pre_image_name:
                continue

            geojson_path = annotations_dir / label_name
            pre_image_path = pre_dir / pre_image_name

            if not geojson_path.exists() or not pre_image_path.exists():
                skipped += 1
                continue

            # Output mask name matches the tile grid ID
            mask_stem = label_name.replace(".geojson", "")
            mask_path = output_dir / f"{mask_stem}_mask.tif"

            try:
                generate_mask(
                    geojson_path=geojson_path,
                    reference_image_path=pre_image_path,
                    reference_csv_path=ref_csv,
                    output_path=mask_path,
                    road_buffer_pixels=road_buffer_pixels,
                )
                generated.append(mask_path)
            except Exception as e:
                logger.error(f"Failed to generate mask for {label_name}: {e}")
                skipped += 1

    logger.info(
        f"Generated {len(generated)} masks, skipped {skipped} "
        f"in {dataset_dir.name}"
    )
    return generated
