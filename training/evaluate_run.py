"""
Final Evaluation of a Trained Run.

Runs once after training, with the best (EMA) weights, and is resumable at
tile granularity:

1. Thresholds: per-head probability thresholds maximizing pixel F1 on the
   validation tiles (flood heads over labelled objects only).
2. Test: every test tile is predicted (sliding window, flip TTA), vectorized
   into buildings and a road graph, and scored: building detection and flood
   attribution F1, SpaceNet 8 building flood IoU, APLS (all roads, and the
   traversable / flooded networks), road and flooded-road length F1, pixel
   IoU. Per-tile records go to tiles_test.jsonl; aggregates and tile-level
   bootstrap 95% confidence intervals to metrics_test.json.
3. Stress test: the post image of a fixed subset of test tiles is shifted
   by 0-40 px to measure robustness to residual misalignment.
4. Timing: seconds to map one 1300 x 1300 tile.

The non-traversable (flooded) APLS is averaged only over tiles that contain
flooded reference roads: on other tiles it is 1.0 by construction.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from dataloader.prepare import load_tile
from evaluation import (
    aggregate_building_counts,
    aggregate_road_lengths,
    apls,
    apls_flood,
    bootstrap_ci,
    building_metrics,
    class_pixel_counts,
    flooded_road_f1,
    mean_defined,
    pixel_metrics,
    road_length_f1,
    shift_image,
    sum_counts,
)
from inference.postprocess import vectorize_prediction
from training.engine import write_json
from training.inference import predict_tile, to_binary

logger = logging.getLogger(__name__)

TUNED_HEADS = ("building", "road", "flooded_building", "flooded_road")
PIXEL_CLASSES = ("building", "road", "flooded_building", "flooded_road")
TILE_MARGIN_S = 120


def _time_left(deadline: Optional[float]) -> float:
    return float("inf") if deadline is None else deadline - time.time()


def _predict(model, tile, cfg, device, tta: bool, post=None) -> dict:
    ev = cfg["eval"]
    return predict_tile(
        model, tile["pre"], tile["post"] if post is None else post,
        window=ev["window"], overlap=ev["overlap"], batch_size=ev["window_batch"],
        device=device, amp=cfg.get("amp", True), tta=tta,
    )


def _load_state(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


# ─── 1. Thresholds ──────────────────────────────────────────────────────────


def tune_thresholds(model, cfg, processed_dir, run_dir, val_ids, deadline, device) -> Optional[dict]:
    """Grid-search per-head thresholds on validation tiles; None if paused."""
    state_path = run_dir / "eval_thresholds_state.json"
    state = _load_state(state_path) or {"done": [], "counts": {}}
    grid = cfg["eval"]["threshold_grid"]

    for tile_id in val_ids:
        if tile_id in state["done"]:
            continue
        if _time_left(deadline) < TILE_MARGIN_S:
            write_json(state_path, state)
            return None
        tile = load_tile(processed_dir, tile_id, cfg.get("register_post", False))
        probs = _predict(model, tile, cfg, device, tta=cfg["eval"].get("tta", True))
        t = tile["targets"]
        masks = {
            "building": t["valid"], "road": t["valid"],
            "flooded_building": t["valid"] & t["building"], "flooded_road": t["valid"] & t["road"],
        }
        for head in TUNED_HEADS:
            target = t[head][masks[head]]
            p = probs[head][masks[head]]
            for thr in grid:
                pred = p > thr
                c = state["counts"].setdefault(f"{head}@{thr}", [0, 0, 0])
                c[0] += int((pred & target).sum())
                c[1] += int((pred & ~target).sum())
                c[2] += int((~pred & target).sum())
        state["done"].append(tile_id)

    thresholds, f1s = {}, {}
    for head in TUNED_HEADS:
        best_f1, best_t = -1.0, 0.5
        for thr in grid:
            tp, fp, fn = state["counts"].get(f"{head}@{thr}", [0, 0, 0])
            f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, thr
        thresholds[head], f1s[head] = best_t, round(best_f1, 4)
    write_json(run_dir / "metrics_val.json", {
        "thresholds": thresholds, "val_pixel_f1_at_threshold": f1s, "n_tiles": len(state["done"]),
    })
    return thresholds


# ─── 2. Test ────────────────────────────────────────────────────────────────


def score_tile(probs: dict, tile: dict, gt: dict, thresholds: dict) -> dict:
    """Vectorize one prediction and compute every per-tile metric."""
    gsd = tile["meta"]["gsd_m"]
    t = tile["targets"]
    binary = to_binary(probs, thresholds)
    vec = vectorize_prediction({"binary_mask": binary}, gsd_m=gsd)

    b = building_metrics(vec["buildings"], gt["buildings"], valid=t["valid"])
    road = road_length_f1(vec["roads"], gt["roads"], gsd)
    flooded_road = flooded_road_f1(vec["roads"], gt["roads"], gsd)
    af = apls_flood(gt["roads"], vec["graph"], gsd, max_pairs=None)
    pred_stack = binary[[0, 1, 3, 4]].astype(bool)
    target_stack = np.stack([t[k] for k in PIXEL_CLASSES])
    return {
        "id": tile["meta"]["id"],
        "gsd_m": gsd,
        "building_counts": b["counts"],
        "road_lengths": {k: road[k] for k in ("gt_m", "pred_m", "gt_matched_m", "pred_matched_m")},
        "flooded_road_lengths": {k: flooded_road[k] for k in ("gt_m", "pred_m", "gt_matched_m", "pred_matched_m")},
        "apls": apls(gt["roads"], vec["graph"], gsd, max_pairs=None),
        **{k: af[k] for k in ("apls_traversable", "apls_non_traversable", "gt_traversable_m",
                              "gt_non_traversable_m", "pred_non_traversable_m")},
        "pixel_counts": class_pixel_counts(pred_stack, target_stack, PIXEL_CLASSES, t["valid"]),
        "flood_image_prob": float(probs["flood_image"]),
        "has_flood": bool(tile["meta"]["has_flood"]),
        "n_pred_buildings": len(vec["buildings"]["features"]),
        "pred_road_km": vec["summary"]["road_km"],
    }


def aggregate(records: list[dict]) -> dict:
    """Micro-averaged metrics and bootstrap CIs over per-tile records."""
    bld = lambda rs: aggregate_building_counts([r["building_counts"] for r in rs])["metrics"]
    road = lambda rs: aggregate_road_lengths([r["road_lengths"] for r in rs])["f1"]
    froad = lambda rs: aggregate_road_lengths([r["flooded_road_lengths"] for r in rs])["f1"]
    with_roads = [r for r in records if r["gt_traversable_m"] + r["gt_non_traversable_m"] > 0]
    with_flooded_roads = [r for r in records if r["gt_non_traversable_m"] > 0]
    apls_mean = lambda rs: mean_defined([r["apls"] for r in rs])
    apls_trav = lambda rs: mean_defined([r["apls_traversable"] for r in rs if r["gt_traversable_m"] > 0])
    apls_flood_net = lambda rs: mean_defined([r["apls_non_traversable"] for r in rs if r["gt_non_traversable_m"] > 0])

    building = bld(records)
    metrics = {
        "building": building,
        "road_length": aggregate_road_lengths([r["road_lengths"] for r in records]),
        "flooded_road_length": aggregate_road_lengths([r["flooded_road_lengths"] for r in records]),
        "apls": apls_mean(with_roads),
        "apls_traversable": apls_trav(records),
        "apls_flooded_network": apls_flood_net(records),
        "pixel": pixel_metrics(sum_counts([r["pixel_counts"] for r in records])),
        "n_tiles": len(records),
        "n_tiles_with_roads": len(with_roads),
        "n_tiles_with_flooded_roads": len(with_flooded_roads),
    }
    sn8_road = mean_defined([metrics["apls_traversable"], metrics["apls_flooded_network"]])
    metrics["sn8_road_score"] = sn8_road

    ci_specs = {
        "building_f1": (records, lambda rs: bld(rs)["all"]["f1"]),
        "flooded_building_f1": (records, lambda rs: bld(rs)["flooded"]["f1"]),
        "building_flood_iou": (records, lambda rs: bld(rs).get("flood_iou")),
        "apls": (with_roads, apls_mean),
        "apls_flooded_network": (with_flooded_roads, apls_flood_net),
        "road_f1": (records, road),
        "flooded_road_f1": (records, froad),
    }
    ci = {name: bootstrap_ci(rs, fn, n=1000, seed=0) for name, (rs, fn) in ci_specs.items() if rs}

    summary = {
        "building_f1": building["all"]["f1"],
        "flooded_building_f1": building["flooded"]["f1"],
        "building_flood_iou": building.get("flood_iou"),
        "apls": metrics["apls"],
        "apls_flooded_network": metrics["apls_flooded_network"],
        "flooded_road_f1": metrics["flooded_road_length"]["f1"],
    }
    return {"summary": {k: v for k, v in summary.items() if v is not None}, "metrics": metrics, "ci": ci}


def evaluate_test(model, cfg, processed_dir, run_dir, test_ids, thresholds, deadline, device) -> Optional[dict]:
    """Score every test tile (resumable); None if paused."""
    records_path = run_dir / "tiles_test.jsonl"
    records = [json.loads(l) for l in records_path.read_text().splitlines()] if records_path.exists() else []
    done = {r["id"] for r in records}

    for tile_id in test_ids:
        if tile_id in done:
            continue
        if _time_left(deadline) < TILE_MARGIN_S:
            return None
        tile = load_tile(processed_dir, tile_id, cfg.get("register_post", False))
        gt = json.loads((Path(processed_dir) / "tiles" / tile_id / "gt.json").read_text())
        t0 = time.time()
        probs = _predict(model, tile, cfg, device, tta=cfg["eval"].get("tta", True))
        record = score_tile(probs, tile, gt, thresholds)
        record["seconds"] = round(time.time() - t0, 2)
        with open(records_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        records.append(record)

    result = aggregate(records)
    result["thresholds"] = thresholds
    write_json(run_dir / "metrics_test.json", result)
    return result


# ─── 3. Stress test and 4. timing ───────────────────────────────────────────


def stress_test(model, cfg, processed_dir, run_dir, test_ids, thresholds, deadline, device) -> bool:
    """Accuracy vs synthetic post-image shift; False if paused."""
    path = run_dir / "stress_partial.json"
    partial = _load_state(path)
    tiles = sorted(test_ids)[: cfg["eval"]["stress_tiles"]]

    for shift in cfg["eval"]["stress_shifts_px"]:
        key = str(shift)
        partial.setdefault(key, {})
        d = int(round(shift / np.sqrt(2)))  # diagonal shift of the requested magnitude
        for tile_id in tiles:
            if tile_id in partial[key]:
                continue
            if _time_left(deadline) < TILE_MARGIN_S:
                write_json(path, partial)
                return False
            tile = load_tile(processed_dir, tile_id, cfg.get("register_post", False))
            gt = json.loads((Path(processed_dir) / "tiles" / tile_id / "gt.json").read_text())
            probs = _predict(model, tile, cfg, device, tta=False, post=shift_image(tile["post"], d, d))
            partial[key][tile_id] = score_tile(probs, tile, gt, thresholds)
        write_json(path, partial)

    curve = {}
    for key, recs in partial.items():
        agg = aggregate(list(recs.values()))
        curve[key] = {**agg["summary"], "n_tiles": len(recs)}
    write_json(run_dir / "stress_test.json", {"shift_px": curve, "tiles": tiles})
    return True


def timing(model, cfg, processed_dir, run_dir, test_ids, device):
    tiles = sorted(test_ids)[: cfg["eval"]["timing_tiles"]]
    seconds = []
    for tile_id in tiles:
        tile = load_tile(processed_dir, tile_id, cfg.get("register_post", False))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        probs = _predict(model, tile, cfg, device, tta=False)
        vectorize_prediction({"binary_mask": to_binary(probs)}, gsd_m=tile["meta"]["gsd_m"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds.append(time.time() - t0)
    write_json(run_dir / "timing.json", {
        "seconds_per_tile_mean": round(float(np.mean(seconds)), 3),
        "seconds_per_tile_all": [round(s, 3) for s in seconds],
        "tile_size_px": 1300,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "includes": "inference (no TTA) + vectorization",
    })


# ─── Entry ──────────────────────────────────────────────────────────────────


def final_evaluation(cfg, model, processed_dir, run_dir, splits, tiles, deadline, device) -> str:
    """Thresholds → test → stress test → timing; returns 'completed' or 'paused'."""
    processed_dir, run_dir = Path(processed_dir), Path(run_dir)
    split = splits[cfg["protocol"]]
    limit = cfg["eval"].get("max_tiles")  # smoke tests only; None = every tile
    val_ids, test_ids = split["val"][:limit], split["test"][:limit]
    model.eval()

    thresholds_path = run_dir / "metrics_val.json"
    if thresholds_path.exists() and "thresholds" in json.loads(thresholds_path.read_text()):
        thresholds = json.loads(thresholds_path.read_text())["thresholds"]
    else:
        thresholds = tune_thresholds(model, cfg, processed_dir, run_dir, val_ids, deadline, device)
        if thresholds is None:
            return "paused"
        logger.info(f"[{cfg['id']}] thresholds {thresholds}")

    if not (run_dir / "metrics_test.json").exists():
        result = evaluate_test(model, cfg, processed_dir, run_dir, test_ids, thresholds, deadline, device)
        if result is None:
            return "paused"
        logger.info(f"[{cfg['id']}] test {result['summary']}")

    if not (run_dir / "stress_test.json").exists():
        if not stress_test(model, cfg, processed_dir, run_dir, test_ids, thresholds, deadline, device):
            return "paused"

    if not (run_dir / "timing.json").exists():
        timing(model, cfg, processed_dir, run_dir, test_ids, device)

    for leftover in ("eval_thresholds_state.json", "stress_partial.json"):
        (run_dir / leftover).unlink(missing_ok=True)
    return "completed"
