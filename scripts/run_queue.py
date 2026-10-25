#!/usr/bin/env python3
"""
Run the experiment queue (one worker per GPU).

Usage:
    # All pending runs of the 'kaggle' queue, 2 GPUs, stop 11.5 h from now
    python scripts/run_queue.py --processed ./processed --results ./results --queue kaggle --gpus 2 --hours 11.5

    # Smoke test: tiny data, a few steps, checkpoint/resume exercised
    python scripts/run_queue.py --processed ./processed_smoke --results ./results_smoke --smoke
"""

import argparse
import copy
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.config import DEFAULT_RUNS_FILE, load_runs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SMOKE_MODELS = ("ra_fusion", "sn8_baseline", "siamese_unet", "snunet", "unet_v0", "siamese_v0")


def smoke_runs(runs: dict) -> list[dict]:
    """One short configuration per model: 2 epochs per stage, small crops."""
    picked = {}
    for cfg in runs.values():
        if cfg["model"] in SMOKE_MODELS and cfg["model"] not in picked and cfg["protocol"] == "P1":
            c = copy.deepcopy(cfg)
            c["id"] = f"smoke_{cfg['model']}"
            c["epochs"] = 2
            c["batch_size"] = 2
            c["crop_size"] = 256
            c["num_workers"] = 0
            c["deadline_margin_s"] = 0
            if c.get("stages"):
                c["stages"] = [{**s, "epochs": 2} for s in c["stages"]]  # a mid-stage resume in each
            c["eval"] = {**c["eval"], "window": 256, "overlap": 32, "stress_tiles": 2, "timing_tiles": 1,
                         "threshold_grid": [0.3, 0.5], "stress_shifts_px": [0, 10], "max_tiles": 6}
            picked[cfg["model"]] = c
    return list(picked.values())


def main():
    parser = argparse.ArgumentParser(description="Run the SpaceNet 8 experiment queue")
    parser.add_argument("--processed", type=str, default="./processed")
    parser.add_argument("--results", type=str, default="./results")
    parser.add_argument("--runs-file", type=str, default=str(DEFAULT_RUNS_FILE))
    parser.add_argument("--queue", type=str, default=None, help="Only runs with this queue name")
    parser.add_argument("--only", type=str, nargs="*", default=None, help="Only these run ids")
    parser.add_argument("--gpus", type=int, default=1, help="Parallel workers, one per GPU")
    parser.add_argument("--hours", type=float, default=None, help="Session time budget")
    parser.add_argument("--deadline", type=float, default=None, help="Absolute deadline (unix time)")
    parser.add_argument("--smoke", action="store_true", help="Short runs of every model + resume test")
    parser.add_argument("--worker", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--session", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    from training.queue import run_worker, write_progress

    runs = load_runs(args.runs_file)
    if args.smoke:
        selected = smoke_runs(runs)
    else:
        selected = [c for c in runs.values()
                    if (args.queue is None or c.get("queue") == args.queue)
                    and (args.only is None or c["id"] in args.only)]
    deadline = args.deadline or (time.time() + args.hours * 3600 if args.hours else None)
    session = args.session or uuid.uuid4().hex[:8]
    max_steps = 3 if args.smoke else None

    if args.worker is not None:
        run_worker(selected, args.processed, args.results, deadline, session, args.worker, max_steps)
        return

    if args.smoke:
        # Checkpoint/resume test: stop every run after one epoch, then resume it to completion
        first = run_worker(selected, args.processed, args.results, deadline, session, "0", max_steps,
                           stop_after_epochs=1)
        logger.info(f"Smoke pass 1 (stop after 1 epoch): {first}")
        while True:
            outcome = run_worker(selected, args.processed, args.results, deadline, session, "0", max_steps,
                                 stop_after_epochs=1)
            if not outcome:
                break
        write_progress(selected, args.results)
        return

    logger.info(f"{len(selected)} runs selected · {args.gpus} worker(s) · session {session}")
    if args.gpus <= 1:
        run_worker(selected, args.processed, args.results, deadline, session, "0")
    else:
        procs = []
        for gpu in range(args.gpus):
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            cmd = [sys.executable, __file__, "--processed", args.processed, "--results", args.results,
                   "--runs-file", args.runs_file, "--worker", str(gpu), "--session", session]
            if args.queue:
                cmd += ["--queue", args.queue]
            if args.only:
                cmd += ["--only", *args.only]
            if deadline:
                cmd += ["--deadline", str(deadline)]
            procs.append(subprocess.Popen(cmd, env=env))
            time.sleep(20)  # stagger startup (model download, cache warm-up)
        codes = [p.wait() for p in procs]
        logger.info(f"Workers finished with exit codes {codes}")
    write_progress(selected, args.results)


if __name__ == "__main__":
    main()
