#!/usr/bin/env python3
"""
Measure GPU memory and speed of every model at its real training settings.

For each distinct model configuration in the registry, runs a few AMP
training steps on random data at the configured crop and batch size, then
reports peak memory, seconds per step and the projected epoch / run time. A
configuration that runs out of memory is retried at half the batch size and
the largest batch that fits is recommended.

Usage:
    python scripts/gpu_probe.py --output results/gpu_probe.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import build_research_model
from training.config import DEFAULT_RUNS_FILE, load_runs
from training.losses import MultiTaskLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

TRAIN_TILES = {"P1": 479, "P2": 181, "P3": 494}


def probe(cfg: dict, batch_size: int, steps: int, device: torch.device) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_research_model(cfg["model"], **cfg.get("model_kwargs", {})).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    criterion = MultiTaskLoss()
    c = cfg["crop_size"]
    has_align = getattr(model, "aligner", None) is not None

    def batch():
        labels = (torch.rand(batch_size, 6, c, c, device=device) > 0.9).float()
        return {
            "pre": torch.rand(batch_size, 3, c, c, device=device),
            "post": torch.rand(batch_size, 3, c, c, device=device),
            "labels": labels,
            "valid": torch.ones(batch_size, 1, c, c, device=device),
            "flood_image": torch.ones(batch_size, 1, device=device),
            "ssl_a": torch.rand(batch_size, 3, c, c, device=device),
            "ssl_b": torch.rand(batch_size, 3, c, c, device=device),
            "ssl_flow": torch.zeros(batch_size, 2, device=device),
            "ssl": torch.ones(batch_size, device=device),
        }

    times = []
    model.train()
    for i in range(steps + 2):
        b = batch()
        torch.cuda.synchronize()
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(b["pre"], b["post"])
            flows = model.estimate_flow(b["ssl_a"][: batch_size // 2], b["ssl_b"][: batch_size // 2]) if has_align else None
            if flows is not None:
                b = {**b, "ssl": torch.cat([torch.ones(batch_size // 2), torch.zeros(batch_size - batch_size // 2)]).to(device)}
            loss = criterion(out, b, flows)["total"]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        if i >= 2:  # skip warm-up steps
            times.append(time.time() - t0)

    sec = sum(times) / len(times)
    steps_per_epoch = cfg.get("samples_per_tile", 2) * TRAIN_TILES[cfg["protocol"]] // batch_size
    epochs = sum(s["epochs"] for s in cfg["stages"]) if cfg.get("stages") else cfg["epochs"]
    del model, optimizer
    return {
        "batch_size": batch_size,
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "sec_per_step": round(sec, 3),
        "epoch_minutes": round(sec * steps_per_epoch / 60, 1),
        "run_hours_train_only": round(sec * steps_per_epoch * epochs / 3600, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Probe GPU memory and speed per model")
    parser.add_argument("--runs-file", type=str, default=str(DEFAULT_RUNS_FILE))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", type=str, default="results/gpu_probe.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("gpu_probe.py needs a CUDA GPU")
    device = torch.device("cuda")

    seen, report = set(), {"gpu": torch.cuda.get_device_name(0), "models": {}}
    for cfg in load_runs(args.runs_file).values():
        key = (cfg["model"], json.dumps(cfg.get("model_kwargs", {}), sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        name = cfg["model"] + ("" if not cfg.get("model_kwargs") else f" {cfg['model_kwargs']}")
        batch = cfg["batch_size"]
        while batch >= 1:
            try:
                result = probe(cfg, batch, args.steps, device)
                result["fits_configured_batch"] = batch == cfg["batch_size"]
                report["models"][name] = result
                logger.info(f"{name}: {result}")
                break
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"{name}: out of memory at batch {batch}")
                torch.cuda.empty_cache()
                batch //= 2
        else:
            report["models"][name] = {"error": "out of memory even at batch 1"}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    logger.info(f"Wrote {out}")


if __name__ == "__main__":
    main()
