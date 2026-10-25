"""
Building Detection and Flood-Attribution Metrics.

Scores predicted building footprints against the reference at the object
level (one-to-one IoU matching, SpaceNet convention IoU >= 0.5) and at the
pixel level (SpaceNet 8 flood IoU: mean pixel IoU over the flooded-building
and non-flooded-building classes).

All functions return raw counts alongside the metrics so tiles can be
aggregated by summing counts (micro-averaging) with
aggregate_building_counts().

Polygons are in pixel [x, y] coordinates of the pre-event tile, the
convention of dataloader.prepare (gt.json) and
inference.postprocess.extract_buildings.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Optional

import numpy as np
import shapely
from shapely.geometry import shape

from evaluation.common import mean_defined, prf_from_counts, safe_div, sum_counts

logger = logging.getLogger(__name__)

IOU_EPS = 1e-9
FLOOD_STATES = (("flooded", True), ("non_flooded", False))


def _load_polygons(fc: Mapping) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse polygon features, repairing invalid geometries.

    Features with empty or zero-area geometry are dropped (with a warning).

    Returns:
        (geometries, flooded flags, original feature indices) as arrays.
    """
    geoms, flooded, index = [], [], []
    for i, feature in enumerate(fc.get("features", [])):
        geom = shape(feature["geometry"])
        if not geom.is_valid:
            geom = shapely.make_valid(geom)
            if geom.geom_type == "GeometryCollection":
                polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
                geom = shapely.unary_union(polys) if polys else shapely.Polygon()
        if geom.is_empty or geom.area <= 0:
            logger.warning(f"Skipping building feature {i}: empty or zero-area geometry")
            continue
        geoms.append(geom)
        flooded.append(bool((feature.get("properties") or {}).get("flooded", False)))
        index.append(i)
    return (
        np.array(geoms, dtype=object),
        np.array(flooded, dtype=bool),
        np.array(index, dtype=np.int64),
    )


def _match(
    pred_geoms: np.ndarray,
    gt_geoms: np.ndarray,
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching by descending IoU over array positions."""
    if len(pred_geoms) == 0 or len(gt_geoms) == 0:
        return []
    tree = shapely.STRtree(gt_geoms)
    pi, gi = tree.query(pred_geoms, predicate="intersects")
    if len(pi) == 0:
        return []
    inter = shapely.area(shapely.intersection(pred_geoms[pi], gt_geoms[gi]))
    union = shapely.area(pred_geoms[pi]) + shapely.area(gt_geoms[gi]) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)

    keep = (iou >= iou_threshold - IOU_EPS) & (iou > 0)
    pi, gi, iou = pi[keep], gi[keep], iou[keep]
    order = np.lexsort((gi, pi, -iou))

    used_pred, used_gt, matches = set(), set(), []
    for k in order:
        p, g = int(pi[k]), int(gi[k])
        if p in used_pred or g in used_gt:
            continue
        used_pred.add(p)
        used_gt.add(g)
        matches.append((p, g, float(iou[k])))
    return matches


def match_buildings(pred_fc: Mapping, gt_fc: Mapping, iou_threshold: float = 0.5) -> dict:
    """
    Match predicted to reference buildings one-to-one by IoU.

    Candidate pairs (intersecting footprints, found with an STRtree) are
    visited in order of decreasing IoU and accepted while both buildings are
    still unmatched and IoU >= iou_threshold (a 1e-9 tolerance absorbs
    floating-point error at the threshold). Ties are broken by feature index,
    so the result is deterministic.

    Args:
        pred_fc: Predicted building FeatureCollection.
        gt_fc: Reference building FeatureCollection.
        iou_threshold: Minimum IoU of a match.

    Returns:
        Dict with 'matches' (list of (pred_index, gt_index, iou), indices into
        the input feature lists, sorted by decreasing IoU), 'unmatched_pred'
        and 'unmatched_gt' (sorted feature indices). Features with empty or
        zero-area geometry appear in neither list.
    """
    pred_geoms, _, pred_idx = _load_polygons(pred_fc)
    gt_geoms, _, gt_idx = _load_polygons(gt_fc)
    matches = _match(pred_geoms, gt_geoms, iou_threshold)
    matched_p = {p for p, _, _ in matches}
    matched_g = {g for _, g, _ in matches}
    return {
        "matches": [(int(pred_idx[p]), int(gt_idx[g]), iou) for p, g, iou in matches],
        "unmatched_pred": [int(pred_idx[p]) for p in range(len(pred_idx)) if p not in matched_p],
        "unmatched_gt": [int(gt_idx[g]) for g in range(len(gt_idx)) if g not in matched_g],
    }


def _state_raster(geoms: np.ndarray, flooded: np.ndarray, tile_shape: tuple[int, int]) -> np.ndarray:
    """Rasterize footprints as 0 = background, 1 = non-flooded, 2 = flooded."""
    from rasterio.features import rasterize

    shapes = [(g, 1) for g, f in zip(geoms, flooded) if not f]
    shapes += [(g, 2) for g, f in zip(geoms, flooded) if f]
    if not shapes:
        return np.zeros(tile_shape, dtype=np.uint8)
    return rasterize(shapes, out_shape=tile_shape, fill=0, dtype="uint8")


def _class_counts(pred: np.ndarray, gt: np.ndarray, value: int, valid: Optional[np.ndarray]) -> dict:
    """Pixel tp/fp/fn of one label value."""
    p, g = pred == value, gt == value
    if valid is not None:
        p, g = p & valid, g & valid
    tp = int(np.count_nonzero(p & g))
    return {"tp": tp, "fp": int(np.count_nonzero(p)) - tp, "fn": int(np.count_nonzero(g)) - tp}


def building_metrics_from_counts(counts: Mapping) -> dict:
    """
    Derive building metrics from (possibly aggregated) counts.

    Args:
        counts: The 'counts' dict of building_metrics() or a sum of them.

    Returns:
        Dict with 'all', 'flooded', 'non_flooded' (each precision / recall /
        f1 / iou), 'state_accuracy' (share of matched buildings whose flood
        state is correct) and, when pixel counts are present,
        'pixel_iou_flooded', 'pixel_iou_non_flooded' and 'flood_iou' (their
        mean over the defined classes). Undefined values are None.
    """
    metrics = {
        name: prf_from_counts(counts[name]["tp"], counts[name]["fp"], counts[name]["fn"])
        for name in ("all", "flooded", "non_flooded")
    }
    metrics["state_accuracy"] = safe_div(counts["state_correct"], counts["all"]["tp"])
    if "pixel" in counts:
        iou = {
            name: prf_from_counts(c["tp"], c["fp"], c["fn"])["iou"]
            for name, c in counts["pixel"].items()
        }
        metrics["pixel_iou_flooded"] = iou["flooded"]
        metrics["pixel_iou_non_flooded"] = iou["non_flooded"]
        metrics["flood_iou"] = mean_defined([iou["flooded"], iou["non_flooded"]])
    return metrics


def building_metrics(
    pred_fc: Mapping,
    gt_fc: Mapping,
    tile_shape: Optional[tuple[int, int]] = None,
    iou_threshold: float = 0.5,
    valid: Optional[np.ndarray] = None,
) -> dict:
    """
    Object- and pixel-level building metrics for one tile.

    Detection ('all'): a prediction is a true positive when matched to a
    reference building (match_buildings). Flood attribution ('flooded',
    'non_flooded'): for each state, a true positive is a matched pair in
    which both buildings have that state; every other prediction with that
    state is a false positive and every other reference building with that
    state a false negative. A match with the wrong state therefore costs one
    false positive in the predicted state and one false negative in the
    reference state.

    Pixel flood IoU (SpaceNet 8 style) rasterizes predicted and reference
    footprints by state on the tile grid (pixel centers inside the polygon;
    flooded wins where footprints of different states overlap) and computes
    the IoU of the flooded and non-flooded classes.

    Args:
        pred_fc: Predicted building FeatureCollection (properties.flooded).
        gt_fc: Reference building FeatureCollection (properties.flooded).
        tile_shape: (H, W) of the tile grid; pixel metrics are skipped when
            neither this nor valid is given.
        iou_threshold: Minimum IoU of an object match.
        valid: Optional boolean (H, W) mask restricting the pixel metrics.

    Returns:
        Dict with 'counts' ('all', 'flooded', 'non_flooded' each {'tp',
        'fp', 'fn'}; 'state_correct'; 'pixel' per-class pixel counts when
        computed) and 'metrics' (building_metrics_from_counts()).
    """
    pred_geoms, pred_flooded, _ = _load_polygons(pred_fc)
    gt_geoms, gt_flooded, _ = _load_polygons(gt_fc)
    matches = _match(pred_geoms, gt_geoms, iou_threshold)

    n_tp = len(matches)
    counts = {"all": {"tp": n_tp, "fp": len(pred_geoms) - n_tp, "fn": len(gt_geoms) - n_tp}}
    for name, state in FLOOD_STATES:
        tp = sum(1 for p, g, _ in matches if pred_flooded[p] == state and gt_flooded[g] == state)
        counts[name] = {
            "tp": tp,
            "fp": int(np.count_nonzero(pred_flooded == state)) - tp,
            "fn": int(np.count_nonzero(gt_flooded == state)) - tp,
        }
    counts["state_correct"] = sum(1 for p, g, _ in matches if pred_flooded[p] == gt_flooded[g])

    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if tile_shape is not None and tuple(tile_shape) != valid.shape:
            raise ValueError(f"tile_shape {tuple(tile_shape)} does not match valid {valid.shape}")
        tile_shape = valid.shape
    if tile_shape is not None:
        tile_shape = (int(tile_shape[0]), int(tile_shape[1]))
        pred_r = _state_raster(pred_geoms, pred_flooded, tile_shape)
        gt_r = _state_raster(gt_geoms, gt_flooded, tile_shape)
        counts["pixel"] = {
            "flooded": _class_counts(pred_r, gt_r, 2, valid),
            "non_flooded": _class_counts(pred_r, gt_r, 1, valid),
        }

    return {"counts": counts, "metrics": building_metrics_from_counts(counts)}


def aggregate_building_counts(results: Sequence[Mapping]) -> dict:
    """
    Micro-average building metrics over tiles by summing their counts.

    Args:
        results: building_metrics() outputs (or their 'counts' dicts).

    Returns:
        Dict with summed 'counts', 'metrics' recomputed from them and
        'n_tiles'.
    """
    counts = [r["counts"] if "counts" in r else r for r in results]
    with_pixel = sum(1 for c in counts if "pixel" in c)
    if 0 < with_pixel < len(counts):
        raise ValueError("Pixel counts present for some tiles only; pass tile_shape for every tile")
    if not counts:
        empty = {"tp": 0, "fp": 0, "fn": 0}
        total = {"all": dict(empty), "flooded": dict(empty), "non_flooded": dict(empty), "state_correct": 0}
    else:
        total = sum_counts(counts)
    return {"counts": total, "metrics": building_metrics_from_counts(total), "n_tiles": len(counts)}
