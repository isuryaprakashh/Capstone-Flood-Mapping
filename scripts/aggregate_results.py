#!/usr/bin/env python3
"""
Aggregate completed runs into the paper's tables.

Runs that differ only in their seed (ids ending in _s<N>) form one group,
reported as mean ± std over seeds; single runs report their 95 % bootstrap CI.

Usage:
    python scripts/aggregate_results.py --results ./results --output ./results/RESULTS.md
"""

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

import yaml

METRICS = [
    ("building_f1", "Building F1"),
    ("flooded_building_f1", "Flooded bldg F1"),
    ("building_flood_iou", "Bldg flood IoU"),
    ("apls", "APLS"),
    ("apls_flooded_network", "APLS flooded"),
    ("flooded_road_f1", "Flooded road F1"),
]
STRESS_METRICS = ("flooded_building_f1", "flooded_road_f1", "building_flood_iou", "building_f1", "apls")
# Tile rule of current stress tests, as in training/evaluate_run.py (not imported: it needs torch)
STRESS_SELECTION = "flooded_first"


def load_completed(results_dir: Path) -> list[dict]:
    runs = []
    for status_path in sorted(results_dir.glob("*/status.json")):
        run_dir = status_path.parent
        status = json.loads(status_path.read_text())
        if status.get("state") != "completed" or not (run_dir / "metrics_test.json").exists():
            continue
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        runs.append({
            "id": run_dir.name,
            "group": re.sub(r"_s\d+$", "", run_dir.name),
            "model": cfg["model"],
            "protocol": cfg["protocol"],
            "seed": cfg["seed"],
            "test": json.loads((run_dir / "metrics_test.json").read_text()),
            "stress": _read_json(run_dir / "stress_test.json"),
            "timing": _read_json(run_dir / "timing.json"),
        })
    return runs


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def summarize(group: list[dict]) -> dict:
    """Per metric: mean and std over seeds, or the bootstrap CI for a single run."""
    out = {}
    for key, _ in METRICS:
        values = [r["test"]["summary"][key] for r in group if r["test"]["summary"].get(key) is not None]
        if not values:
            out[key] = None
        elif len(values) > 1:
            out[key] = {"mean": statistics.mean(values), "std": statistics.stdev(values), "n": len(values)}
        else:
            ci = group[0]["test"].get("ci", {}).get(key) or {}
            out[key] = {"mean": values[0], "low": ci.get("low"), "high": ci.get("high"), "n": 1}
    return out


def fmt(cell) -> str:
    if cell is None:
        return "–"
    if cell["n"] > 1:
        return f"{cell['mean']:.3f} ± {cell['std']:.3f}"
    if cell.get("low") is not None:
        return f"{cell['mean']:.3f} [{cell['low']:.3f}, {cell['high']:.3f}]"
    return f"{cell['mean']:.3f}"


def main():
    parser = argparse.ArgumentParser(description="Aggregate completed runs into tables")
    parser.add_argument("--results", type=str, default="./results")
    parser.add_argument("--output", type=str, default=None, help="Markdown file (default: <results>/RESULTS.md)")
    args = parser.parse_args()

    results_dir = Path(args.results)
    runs = load_completed(results_dir)
    if not runs:
        sys.exit(f"No completed runs in {results_dir}")

    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(r["group"], []).append(r)
    order = sorted(groups, key=lambda g: (groups[g][0]["protocol"], g))

    lines = ["# Results", "",
             "Mean ± std over seeds; single runs show the 95 % bootstrap CI over test tiles.", ""]
    rows = []
    for protocol in sorted({groups[g][0]["protocol"] for g in order}):
        lines += [f"## {protocol}", "",
                  "| Run | Model | Seeds | " + " | ".join(name for _, name in METRICS) + " | s/tile |",
                  "|---|---|---|" + "---|" * (len(METRICS) + 1)]
        for g in (g for g in order if groups[g][0]["protocol"] == protocol):
            group = groups[g]
            summary = summarize(group)
            timings = [r["timing"]["seconds_per_tile_mean"] for r in group if r["timing"]]
            speed = f"{statistics.mean(timings):.2f}" if timings else "–"
            lines.append(f"| {g} | {group[0]['model']} | {len(group)} | "
                         + " | ".join(fmt(summary[k]) for k, _ in METRICS) + f" | {speed} |")
            rows.append({"group": g, "model": group[0]["model"], "protocol": protocol, "seeds": len(group),
                         **{f"{k}_mean": (summary[k] or {}).get("mean") for k, _ in METRICS},
                         **{f"{k}_std": (summary[k] or {}).get("std") for k, _ in METRICS},
                         "seconds_per_tile": statistics.mean(timings) if timings else None})
        lines.append("")

    # Robustness to added pre/post misalignment (first seed of each group). Stress tests made under
    # an older tile rule are left out until their run is re-evaluated.
    stressed = [(g, groups[g][0]["stress"]) for g in order
                if (groups[g][0]["stress"] or {}).get("selection") == STRESS_SELECTION]
    if stressed:
        shifts = sorted({int(s) for _, st in stressed for s in st["shift_px"]})
        for key in STRESS_METRICS:
            lines += [f"## Added misalignment: {key}", "",
                      "| Run | " + " | ".join(f"+{s} px" for s in shifts) + " |",
                      "|---|" + "---|" * len(shifts)]
            for g, st in stressed:
                cells = [st["shift_px"].get(str(s), {}).get(key) for s in shifts]
                lines.append(f"| {g} | " + " | ".join("–" if c is None else f"{c:.3f}" for c in cells) + " |")
            lines.append("")

    output = Path(args.output) if args.output else results_dir / "RESULTS.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    with open(output.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(runs)} runs in {len(groups)} groups -> {output}")


if __name__ == "__main__":
    main()
