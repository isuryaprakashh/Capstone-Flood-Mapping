"""
Ground-Truth Loading for SpaceNet 8 Tiles.

Loads the reference annotations of a labelled tile as pixel-space vectors
(building polygons from the GeoJSON, road centerlines from the reference
CSV) plus the rasterized 5-channel mask, so predictions can be overlaid on
and scored against the reference in the same coordinate frame.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import rasterio
from shapely.affinity import affine_transform
from shapely.geometry import mapping

from dataloader.align import pixel_size_m
from dataloader.mask_generator import (
    generate_mask,
    parse_geojson_annotations,
    parse_reference_csv,
)

logger = logging.getLogger(__name__)


def load_ground_truth(
    geojson_path: Union[str, Path],
    pre_image_path: Union[str, Path],
    reference_csv_path: Optional[Union[str, Path]] = None,
    include_mask: bool = True,
) -> dict:
    """
    Load a tile's reference buildings, roads and rasterized mask.

    Args:
        geojson_path: Tile annotation GeoJSON (geographic coordinates).
        pre_image_path: Pre-event GeoTIFF defining the pixel grid.
        reference_csv_path: AOI reference CSV with pixel-space road WKT.
        include_mask: Also rasterize the 5-channel mask (skip when only the
            vectors are needed).

    Returns:
        Dictionary with 'buildings' and 'roads' (GeoJSON FeatureCollections
        in pixel [x, y] coordinates, each feature with 'flooded'), 'gsd_m'
        and 'mask' ((5, H, W) uint8 in the model's channel order, or None).
    """
    pre_image_path = Path(pre_image_path)
    with rasterio.open(pre_image_path) as src:
        inv = ~src.transform
        gsd_m = pixel_size_m(src.res, src.crs, src.transform)

    ann = parse_geojson_annotations(geojson_path)
    flooded_ids = {id(g) for g in ann["flooded_buildings"]}
    to_pixel = [inv.a, inv.b, inv.d, inv.e, inv.xoff, inv.yoff]

    buildings = []
    for poly in ann["buildings"]:
        flooded = id(poly) in flooded_ids
        px = affine_transform(poly, to_pixel)
        buildings.append({
            "type": "Feature",
            "geometry": mapping(px),
            "properties": {
                "flooded": flooded,
                "status": "flooded" if flooded else "intact",
                "area_m2": round(float(px.area) * gsd_m ** 2, 1),
            },
        })

    roads = []
    if reference_csv_path and Path(reference_csv_path).exists():
        ref = parse_reference_csv(reference_csv_path, pre_image_path.stem)
        flooded_road_ids = {id(g) for g in ref["flooded_roads_pix"]}
        for line in ref["roads_pix"]:
            flooded = id(line) in flooded_road_ids
            roads.append({
                "type": "Feature",
                "geometry": mapping(line),
                "properties": {
                    "flooded": flooded,
                    "status": "flooded" if flooded else "passable",
                    "length_m": round(float(line.length) * gsd_m, 1),
                },
            })

    mask = None
    if include_mask:
        mask = generate_mask(
            geojson_path=geojson_path,
            reference_image_path=pre_image_path,
            reference_csv_path=reference_csv_path,
        )

    return {
        "buildings": {"type": "FeatureCollection", "features": buildings},
        "roads": {"type": "FeatureCollection", "features": roads},
        "mask": mask,
        "gsd_m": gsd_m,
    }
