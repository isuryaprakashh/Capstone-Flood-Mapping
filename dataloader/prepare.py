"""
SpaceNet 8 Dataset Preparation.

Converts the raw AOI folders into a training-ready layout, once:

  processed/
    manifest.json                      tile list with metadata and statistics
    splits.json                        protocol splits P1/P2/P3 (spatially blocked folds)
    tiles/<aoi>/<tile>/
      pre.png, post.png                RGB, post resampled onto the pre-event grid
      targets.npz                      boolean masks (see TARGET_CHANNELS)
      gt.json                          reference buildings/roads in pixel space
      meta.json                        georeference, chosen post image, alignment

For tiles with two post images, the one with the smallest reliably estimated
residual misalignment is kept. The residual shift is recorded but not applied:
registration is a modelling choice evaluated in the experiments.
"""

import csv
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import rasterio
from rasterio import features
from scipy.ndimage import grey_dilation, grey_erosion
from shapely.geometry import Point, shape

from dataloader.align import apply_shift, estimate_shift_blocks, pixel_size_m, read_rgb, resample_to_grid
from dataloader.ground_truth import load_ground_truth

logger = logging.getLogger(__name__)

LABELLED_AOIS = ("Germany_Training_Public", "Louisiana-East_Training_Public")
UNLABELLED_AOIS = ("Louisiana-West_Test_Public",)

TARGET_CHANNELS = (
    "building",          # building footprints
    "building_border",   # instance boundaries (separates touching buildings)
    "road",              # buffered road centerlines
    "junction",          # disks around road intersections (connectivity cue)
    "flooded_building",  # buildings attributed flooded
    "flooded_road",      # roads attributed flooded
    "valid",             # pixels covered by both pre and post imagery
)

ROAD_HALF_WIDTH_M = 2.0
JUNCTION_RADIUS_M = 6.0
MAX_RELIABLE_SHIFT_PX = 80.0


# ─── Discovery ──────────────────────────────────────────────────────────────


def list_tiles(data_root: Union[str, Path], include_unlabelled: bool = False) -> list[dict]:
    """
    List every tile from the AOI mapping CSVs.

    Args:
        data_root: Folder containing the AOI folders.
        include_unlabelled: Also list the unlabelled test AOI.

    Returns:
        One dict per tile with paths to the pre image, post image(s),
        annotation GeoJSON and reference CSV (None when unlabelled).
    """
    data_root = Path(data_root)
    aois = LABELLED_AOIS + (UNLABELLED_AOIS if include_unlabelled else ())
    tiles = []
    for aoi in aois:
        aoi_dir = data_root / aoi
        mapping = next(aoi_dir.glob("*_label_image_mapping.csv"), None)
        if mapping is None:
            logger.warning(f"No mapping CSV in {aoi_dir}")
            continue
        reference = next(aoi_dir.glob("*_reference.csv"), None)
        labelled = (aoi_dir / "annotations").exists()

        with open(mapping) as f:
            for row in csv.DictReader(f):
                label = (row.get("label") or "").strip()
                pre = (row.get("pre-event image") or "").strip()
                posts = [
                    (row.get(k) or "").strip()
                    for k in ("post-event image 1", "post-event image 2")
                ]
                posts = [p for p in posts if p]
                if not pre or not posts:
                    continue
                geojson = aoi_dir / "annotations" / label if labelled else None
                tiles.append({
                    "aoi": aoi,
                    "tile": label.replace(".geojson", "") or Path(pre).stem,
                    "pre_path": str(aoi_dir / "PRE-event" / pre),
                    "post_paths": [str(aoi_dir / "POST-event" / p) for p in posts],
                    "geojson_path": str(geojson) if geojson and geojson.exists() else None,
                    "reference_csv": str(reference) if reference else None,
                })
    return tiles


# ─── Imagery ────────────────────────────────────────────────────────────────


def read_pre(path: Union[str, Path]) -> tuple[np.ndarray, dict, np.ndarray]:
    """Read the pre-event image, its georeference and valid-data mask."""
    rgb, geo = read_rgb(path)
    with rasterio.open(path) as src:
        valid = src.dataset_mask() > 0  # respects the 255 nodata value of SN8 pre tiles
    return rgb, geo, valid


def choose_post(pre: np.ndarray, pre_geo: dict, pre_valid: np.ndarray, post_paths: list[str]) -> dict:
    """
    Resample each candidate post image onto the pre grid and keep the best.

    Ranking: a reliable residual-shift estimate beats an unreliable one, then
    smaller residual shift, then larger valid coverage.

    Returns:
        Dict with 'index', 'name', 'post' (3, H, W), 'valid' (H, W) and
        'candidates' (alignment statistics of every post image).
    """
    candidates = []
    for idx, path in enumerate(post_paths):
        post, post_valid = resample_to_grid(path, pre_geo)
        both = post_valid & pre_valid
        reg = estimate_shift_blocks(pre, post, both)
        reliable = reg["n_confident"] >= 2 and reg["magnitude"] <= MAX_RELIABLE_SHIFT_PX
        candidates.append({
            "index": idx,
            "name": Path(path).name,
            "post": post,
            "valid": post_valid,
            "stats": {
                "name": Path(path).name,
                "shift_dy_px": None if not reliable else round(reg["dy"], 2),
                "shift_dx_px": None if not reliable else round(reg["dx"], 2),
                "shift_px": None if not reliable else round(reg["magnitude"], 2),
                "spread_px": None if not reliable else round(reg["spread"], 2),
                "confident_blocks": reg["n_confident"],
                "coverage": round(float(both.mean()), 4),
                "reliable": reliable,
            },
        })

    best = min(
        candidates,
        key=lambda c: (
            not c["stats"]["reliable"],
            c["stats"]["shift_px"] if c["stats"]["reliable"] else 0.0,
            -c["stats"]["coverage"],
        ),
    )
    return {
        "index": best["index"],
        "name": best["name"],
        "post": best["post"],
        "valid": best["valid"],
        "candidates": [c["stats"] for c in candidates],
    }


# ─── Targets ────────────────────────────────────────────────────────────────


def road_junctions(roads: dict, tol_px: float = 1.0) -> list[tuple[float, float]]:
    """
    Find road intersections in reference centerlines.

    A segment endpoint is a junction when at least three road ends meet there;
    an endpoint touching another segment's interior counts that segment twice
    (it continues through the point).

    Returns:
        List of (x, y) pixel positions.
    """
    lines = [shape(f["geometry"]) for f in roads.get("features", [])]
    lines = [l for l in lines if not l.is_empty and l.geom_type == "LineString"]
    ends = [Point(c) for l in lines for c in (l.coords[0], l.coords[-1])]

    junctions = []
    for p in ends:
        if any(p.distance(Point(j)) <= tol_px for j in junctions):
            continue
        degree = 0
        for l in lines:
            if p.distance(Point(l.coords[0])) <= tol_px or p.distance(Point(l.coords[-1])) <= tol_px:
                degree += 1
            elif l.distance(p) <= tol_px:
                degree += 2
        if degree >= 3:
            junctions.append((p.x, p.y))
    return junctions


def rasterize_targets(
    shape_hw: tuple[int, int],
    buildings: dict,
    roads: dict,
    gsd_m: float,
) -> dict[str, np.ndarray]:
    """
    Rasterize reference vectors into boolean training targets.

    Args:
        shape_hw: (H, W) of the pre-event tile.
        buildings: Building FeatureCollection in pixel coordinates.
        roads: Road FeatureCollection in pixel coordinates.
        gsd_m: Meters per pixel.

    Returns:
        Dict of boolean (H, W) arrays for every channel in TARGET_CHANNELS
        except 'valid'.
    """
    h, w = shape_hw
    b_feats = buildings.get("features", [])
    r_feats = roads.get("features", [])

    instances = np.zeros((h, w), dtype=np.int32)
    if b_feats:
        instances = features.rasterize(
            [(f["geometry"], i + 1) for i, f in enumerate(b_feats)],
            out_shape=(h, w), fill=0, dtype="int32",
        )
    building = instances > 0
    building_border = (grey_dilation(instances, size=3) != grey_erosion(instances, size=3)) & building

    flooded_building = np.zeros((h, w), dtype=bool)
    flooded_shapes = [(f["geometry"], 1) for f in b_feats if f["properties"].get("flooded")]
    if flooded_shapes:
        flooded_building = features.rasterize(flooded_shapes, out_shape=(h, w), fill=0, dtype="uint8") > 0

    half_px = ROAD_HALF_WIDTH_M / gsd_m

    def roads_mask(feats) -> np.ndarray:
        polys = [shape(f["geometry"]).buffer(half_px) for f in feats]
        polys = [(p, 1) for p in polys if not p.is_empty]
        if not polys:
            return np.zeros((h, w), dtype=bool)
        return features.rasterize(polys, out_shape=(h, w), fill=0, dtype="uint8") > 0

    road = roads_mask(r_feats)
    flooded_road = roads_mask([f for f in r_feats if f["properties"].get("flooded")])

    junction = np.zeros((h, w), dtype=np.uint8)
    radius = max(1, int(round(JUNCTION_RADIUS_M / gsd_m)))
    for x, y in road_junctions(roads):
        cv2.circle(junction, (int(round(x)), int(round(y))), radius, 1, thickness=-1)

    return {
        "building": building,
        "building_border": building_border,
        "road": road,
        "junction": junction > 0,
        "flooded_building": flooded_building,
        "flooded_road": flooded_road,
    }


# ─── Per-tile processing ────────────────────────────────────────────────────


def prepare_tile(task: dict) -> Optional[dict]:
    """
    Process one tile and write its files.

    Args:
        task: Entry from list_tiles() plus 'out_dir'.

    Returns:
        Tile metadata (also written to meta.json), or None on failure.
    """
    out = Path(task["out_dir"]) / "tiles" / task["aoi"] / task["tile"]
    try:
        pre, geo, pre_valid = read_pre(task["pre_path"])
        chosen = choose_post(pre, geo, pre_valid, task["post_paths"])
        gsd_m = pixel_size_m(geo["res"], geo["crs"], geo["transform"])
        h, w = pre.shape[1:]

        targets = {name: np.zeros((h, w), dtype=bool) for name in TARGET_CHANNELS}
        buildings = roads = {"type": "FeatureCollection", "features": []}
        labelled = task["geojson_path"] is not None
        if labelled:
            gt = load_ground_truth(task["geojson_path"], task["pre_path"], task["reference_csv"], include_mask=False)
            buildings, roads = gt["buildings"], gt["roads"]
            targets.update(rasterize_targets((h, w), buildings, roads, gsd_m))
        targets["valid"] = pre_valid & chosen["valid"]

        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "pre.png"), np.ascontiguousarray(pre.transpose(1, 2, 0)[:, :, ::-1]))
        cv2.imwrite(str(out / "post.png"), np.ascontiguousarray(chosen["post"].transpose(1, 2, 0)[:, :, ::-1]))
        np.savez_compressed(out / "targets.npz", **targets)
        with open(out / "gt.json", "w") as f:
            json.dump({"buildings": buildings, "roads": roads}, f)

        t = geo["transform"]
        b_props = [f["properties"] for f in buildings["features"]]
        r_props = [f["properties"] for f in roads["features"]]
        meta = {
            "id": f"{task['aoi']}/{task['tile']}",
            "aoi": task["aoi"],
            "tile": task["tile"],
            "labelled": labelled,
            "pre_image": Path(task["pre_path"]).name,
            "post_image": chosen["name"],
            "post_index": chosen["index"],
            "post_candidates": chosen["candidates"],
            "shape": [h, w],
            "gsd_m": round(gsd_m, 4),
            "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
            "crs": str(geo["crs"]),
            "centroid_lonlat": [t.c + t.a * w / 2, t.f + t.e * h / 2],
            "valid_fraction": round(float(targets["valid"].mean()), 4),
            "n_buildings": len(b_props),
            "n_flooded_buildings": sum(1 for p in b_props if p.get("flooded")),
            "road_m": round(sum(p.get("length_m", 0.0) for p in r_props), 1),
            "flooded_road_m": round(sum(p.get("length_m", 0.0) for p in r_props if p.get("flooded")), 1),
        }
        meta["has_flood"] = meta["n_flooded_buildings"] > 0 or meta["flooded_road_m"] > 0
        with open(out / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        return meta

    except Exception as e:
        logger.error(f"Failed to prepare {task['aoi']}/{task['tile']}: {e}")
        return None


# ─── Splits ─────────────────────────────────────────────────────────────────


def assign_folds(
    metas: list[dict],
    n_folds: int = 5,
    block_deg: float = 0.02,
    seed: int = 0,
) -> dict[str, int]:
    """
    Assign tiles to spatially blocked folds, separately per AOI.

    Tiles are grouped into geographic blocks (block_deg x block_deg, ~2 km)
    so neighbouring tiles share a fold. Blocks are assigned greedily to the
    fold with the fewest flooded tiles, then fewest tiles, which balances
    flood prevalence across folds.

    Returns:
        Mapping tile id → fold index.
    """
    rng = np.random.RandomState(seed)
    folds = {}
    for aoi in sorted({m["aoi"] for m in metas}):
        blocks: dict[tuple, list[dict]] = {}
        for m in (m for m in metas if m["aoi"] == aoi):
            lon, lat = m["centroid_lonlat"]
            blocks.setdefault((int(np.floor(lat / block_deg)), int(np.floor(lon / block_deg))), []).append(m)

        keys = sorted(blocks)
        rng.shuffle(keys)
        keys.sort(key=lambda k: (-sum(m["has_flood"] for m in blocks[k]), -len(blocks[k])))

        load = [[0, 0] for _ in range(n_folds)]  # [flooded tiles, tiles]
        for k in keys:
            f = min(range(n_folds), key=lambda i: (load[i][0], load[i][1], i))
            for m in blocks[k]:
                folds[m["id"]] = f
            load[f][0] += sum(m["has_flood"] for m in blocks[k])
            load[f][1] += len(blocks[k])
    return folds


def make_splits(metas: list[dict], folds: dict[str, int]) -> dict:
    """
    Build the evaluation protocols from fold assignments.

    P1: train Louisiana-East (folds 1-4), validate fold 0, test all of Germany.
    P2: train Germany (folds 1-4), validate fold 0, test all of Louisiana-East.
    P3: both AOIs pooled; test fold 0, validate fold 1, train folds 2-4.
    """
    labelled = [m for m in metas if m["labelled"]]
    ger = [m["id"] for m in labelled if m["aoi"] == "Germany_Training_Public"]
    lae = [m["id"] for m in labelled if m["aoi"] == "Louisiana-East_Training_Public"]
    ids = [m["id"] for m in labelled]

    return {
        "P1": {
            "train": [t for t in lae if folds[t] != 0],
            "val": [t for t in lae if folds[t] == 0],
            "test": ger,
        },
        "P2": {
            "train": [t for t in ger if folds[t] != 0],
            "val": [t for t in ger if folds[t] == 0],
            "test": lae,
        },
        "P3": {
            "train": [t for t in ids if folds[t] >= 2],
            "val": [t for t in ids if folds[t] == 1],
            "test": [t for t in ids if folds[t] == 0],
        },
        "unlabelled": [m["id"] for m in metas if not m["labelled"]],
        "folds": folds,
    }


# ─── Driver ─────────────────────────────────────────────────────────────────


def prepare_dataset(
    data_root: Union[str, Path],
    out_dir: Union[str, Path],
    workers: int = 4,
    include_unlabelled: bool = False,
    limit: Optional[int] = None,
    seed: int = 0,
) -> dict:
    """
    Prepare all tiles, then write manifest.json and splits.json.

    Already-prepared tiles (meta.json present) are reused, so an interrupted
    run resumes where it stopped.

    Args:
        data_root: Raw dataset folder (AOI subfolders).
        out_dir: Output folder.
        workers: Parallel processes.
        include_unlabelled: Also prepare the unlabelled test AOI (images only).
        limit: Process at most this many tiles per AOI (smoke tests).
        seed: Seed for the fold assignment.

    Returns:
        The manifest dict.
    """
    out_dir = Path(out_dir)
    tasks = list_tiles(data_root, include_unlabelled)
    if limit:
        kept, per_aoi = [], {}
        for t in tasks:
            if per_aoi.get(t["aoi"], 0) < limit:
                kept.append(t)
                per_aoi[t["aoi"]] = per_aoi.get(t["aoi"], 0) + 1
        tasks = kept

    metas, todo = [], []
    for t in tasks:
        meta_path = out_dir / "tiles" / t["aoi"] / t["tile"] / "meta.json"
        if meta_path.exists():
            metas.append(json.loads(meta_path.read_text()))
        else:
            todo.append({**t, "out_dir": str(out_dir)})

    logger.info(f"Preparing {len(todo)} tiles ({len(metas)} already done) with {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, meta in enumerate(pool.map(prepare_tile, todo, chunksize=2), start=1):
            if meta is not None:
                metas.append(meta)
            if i % 50 == 0 or i == len(todo):
                logger.info(f"  {i}/{len(todo)} tiles")

    metas.sort(key=lambda m: m["id"])
    folds = assign_folds([m for m in metas if m["labelled"]], seed=seed)
    splits = make_splits(metas, folds)

    manifest = {
        "version": 1,
        "target_channels": list(TARGET_CHANNELS),
        "road_half_width_m": ROAD_HALF_WIDTH_M,
        "junction_radius_m": JUNCTION_RADIUS_M,
        "n_tiles": len(metas),
        "n_failed": len(tasks) - len(metas),
        "tiles": metas,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out_dir / "splits.json").write_text(json.dumps(splits, indent=1))

    for name in ("P1", "P2", "P3"):
        s = splits[name]
        logger.info(f"{name}: train {len(s['train'])} | val {len(s['val'])} | test {len(s['test'])}")
    return manifest


def load_tile(out_dir: Union[str, Path], tile_id: str, register_post: bool = False) -> dict:
    """
    Load a prepared tile.

    Args:
        out_dir: Prepared dataset folder.
        tile_id: '<aoi>/<tile>'.
        register_post: Shift the post image by its stored phase-correlation
            estimate (classical registration) when that estimate is reliable;
            the valid mask is reduced to the shifted coverage.

    Returns:
        Dict with 'pre', 'post' ((3, H, W) uint8 RGB), 'targets' (dict of
        boolean arrays) and 'meta'.
    """
    d = Path(out_dir) / "tiles" / tile_id
    pre = cv2.imread(str(d / "pre.png"), cv2.IMREAD_COLOR)[:, :, ::-1].transpose(2, 0, 1)
    post = cv2.imread(str(d / "post.png"), cv2.IMREAD_COLOR)[:, :, ::-1].transpose(2, 0, 1)
    with np.load(d / "targets.npz") as z:
        targets = {k: z[k] for k in z.files}
    meta = json.loads((d / "meta.json").read_text())

    if register_post:
        est = meta["post_candidates"][meta["post_index"]]
        if est["reliable"]:
            post = apply_shift(np.ascontiguousarray(post), est["shift_dy_px"], est["shift_dx_px"])
            coverage = apply_shift(np.ones((1, *post.shape[1:]), dtype=np.uint8), est["shift_dy_px"], est["shift_dx_px"])[0]
            targets["valid"] = targets["valid"] & (coverage > 0)

    return {"pre": np.ascontiguousarray(pre), "post": np.ascontiguousarray(post), "targets": targets, "meta": meta}
