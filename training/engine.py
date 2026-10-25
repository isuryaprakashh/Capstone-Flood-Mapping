"""
Training Engine with Per-Epoch Checkpoint/Resume.

`train_run` trains one experiment end to end and is safe to call repeatedly:

- After every epoch it writes checkpoints/last.pt (model, EMA weights,
  optimizer, LR schedule, AMP scaler, RNG states, stage/epoch, best score)
  atomically; calling it again resumes from the next epoch.
- A deadline (seconds since the epoch) makes it stop cleanly before a session
  time limit, returning 'paused'; the next call continues.
- Staged training (e.g. the SpaceNet 8 baseline's foundation network, then
  its flood network) restricts trainable parameters and loss terms per stage.
- After training, the final evaluation (threshold tuning on validation, test
  metrics, misalignment stress test) runs once and is itself resumable.

Run folder layout:
    config.yaml, environment.json, status.json, train_log.csv, train.log,
    metrics_val.json, metrics_test.json, stress_test.json, timing.json,
    checkpoints/{last.pt, best.pt}
"""

import csv
import json
import logging
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from models import build_research_model
from training.config import config_hash
from training.dataset import LABEL_CHANNELS, SN8CenterCropDataset, SN8CropDataset, tile_weights
from training.losses import MultiTaskLoss

logger = logging.getLogger(__name__)

SELECTION_HEADS = {
    "foundation": ("building", "road"),
    "flood": ("flooded_building", "flooded_road"),
    "joint": ("building", "road", "flooded_building", "flooded_road"),
}


# ─── Utilities ──────────────────────────────────────────────────────────────


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def environment_info() -> dict:
    """Software/hardware versions recorded with every run."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        commit = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_commit": commit or os.environ.get("CODE_GIT_COMMIT"),
        "kaggle_kernel_run": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
    }


def model_state(model: torch.nn.Module, half: bool = False) -> dict:
    """
    State dict without frozen pretrained weights (model.frozen_prefixes):
    they are reloaded from pretraining, which keeps checkpoints small.
    """
    skip = getattr(model, "frozen_prefixes", ())
    state = {k: v for k, v in model.state_dict().items() if not k.startswith(skip)}
    if half:
        state = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in state.items()}
    return state


def load_model_state(model: torch.nn.Module, state: dict):
    """Load a (possibly frozen-weight-stripped, possibly fp16) state dict."""
    current = model.state_dict()
    missing = [k for k in current if k not in state and not k.startswith(getattr(model, "frozen_prefixes", ()))]
    if missing:
        raise KeyError(f"Checkpoint is missing {len(missing)} weights, e.g. {missing[:3]}")
    model.load_state_dict({**current, **{k: v.to(current[k].dtype) for k, v in state.items()}})


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(path: Path, data: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)


class EMA:
    """Exponential moving average of a model's floating-point state."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        skip = getattr(model, "frozen_prefixes", ())  # frozen weights never change: no shadow copy
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items()
            if v.dtype.is_floating_point and not k.startswith(skip)
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def state_dict(self) -> dict:
        return self.shadow

    def load_state_dict(self, state: dict):
        # Copy into the existing shadow so it stays on the model's device (checkpoints load on CPU)
        for k, v in state.items():
            self.shadow[k].copy_(v)

    def apply_to(self, model: torch.nn.Module) -> dict:
        """Load EMA weights into the model; returns the replaced weights."""
        current = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        model.load_state_dict({**model.state_dict(), **self.shadow})
        return current


def cosine_with_warmup(optimizer, total_steps: int, warmup_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def resolve_stages(cfg: dict) -> list[dict]:
    """
    Stages from the config, or a single joint stage.

    Each stage: name, epochs, heads (loss terms), param_prefixes (None = all
    trainable), init (optional model hook called once when the stage starts).
    """
    if cfg.get("stages"):
        return cfg["stages"]
    return [{"name": "joint", "epochs": cfg["epochs"], "heads": None, "param_prefixes": None, "init": None}]


def stage_loss_weights(cfg: dict, stage: dict) -> dict:
    weights = {k: v for k, v in cfg.get("loss", {}).items() if k != "object_masked_flood"}
    if stage.get("heads"):
        from training.losses import DEFAULT_WEIGHTS
        weights = {k: (weights.get(k, DEFAULT_WEIGHTS[k]) if k in stage["heads"] else 0.0) for k in DEFAULT_WEIGHTS}
    return weights


def set_trainable(model: torch.nn.Module, base: dict[str, bool], prefixes: Optional[list[str]]):
    """Enable gradients for parameters matching `prefixes` (never for frozen-by-design ones)."""
    for name, p in model.named_parameters():
        p.requires_grad = base[name] and (prefixes is None or any(name.startswith(pfx) for pfx in prefixes))


def train_mode(model: torch.nn.Module):
    """model.train(), but normalization layers whose parameters are all frozen keep their statistics."""
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in m.parameters(recurse=False)):
                m.eval()


# ─── Validation ─────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(model, loader, device, amp: bool, heads: tuple[str, ...]) -> dict:
    """Pixel F1 per head on centre crops (flood heads over labelled objects)."""
    model.eval()
    counts = {k: np.zeros(3) for k in heads}  # tp, fp, fn
    use_amp = amp and device.type == "cuda"
    for batch in loader:
        pre, post = batch["pre"].to(device), batch["post"].to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            out = model(pre, post)
        labels, valid = batch["labels"].to(device), batch["valid"].to(device)
        for k in heads:
            i = LABEL_CHANNELS.index(k)
            mask = valid
            if k == "flooded_building":
                mask = valid * labels[:, LABEL_CHANNELS.index("building")][:, None]
            elif k == "flooded_road":
                mask = valid * labels[:, LABEL_CHANNELS.index("road")][:, None]
            pred = (torch.sigmoid(out[k].float()) > 0.5).float() * mask
            target = labels[:, i:i + 1] * mask
            counts[k] += np.array([
                (pred * target).sum().item(),
                (pred * (1 - target)).sum().item(),
                ((1 - pred) * target).sum().item(),
            ])
    metrics = {}
    for k, (tp, fp, fn) in counts.items():
        metrics[f"val_f1_{k}"] = float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn > 0 else 0.0
    metrics["val_score"] = float(np.mean([metrics[f"val_f1_{k}"] for k in heads]))
    return metrics


# ─── Main entry ─────────────────────────────────────────────────────────────


def train_run(
    cfg: dict,
    processed_dir: Union[str, Path],
    run_dir: Union[str, Path],
    deadline: Optional[float] = None,
    device: Optional[torch.device] = None,
    max_steps_per_epoch: Optional[int] = None,
    stop_after_epochs: Optional[int] = None,
) -> str:
    """
    Train (or resume) one run and run its final evaluation.

    Args:
        cfg: Resolved run configuration (training/config.py).
        processed_dir: Prepared dataset folder.
        run_dir: Output folder of this run.
        deadline: time.time() by which the call must return; the run pauses
            cleanly before it.
        device: Training device (default: CUDA if available).
        max_steps_per_epoch: Cap steps per epoch (smoke tests).
        stop_after_epochs: Pause after this many epochs in this call (used to
            test checkpoint/resume).

    Returns:
        'completed' or 'paused'. Exceptions propagate (the queue records
        them as failures and retries).
    """
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = Path(processed_dir)
    chash = config_hash(cfg)

    file_handler = logging.FileHandler(run_dir / "train.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(file_handler)
    try:
        (run_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "config_hash": chash}, sort_keys=False))
        write_json(run_dir / "environment.json", environment_info())
        return _train(cfg, chash, processed_dir, run_dir, ckpt_dir, deadline, device, max_steps_per_epoch,
                      stop_after_epochs)
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


def _time_left(deadline: Optional[float]) -> float:
    return float("inf") if deadline is None else deadline - time.time()


def _train(cfg, chash, processed_dir, run_dir, ckpt_dir, deadline, device, max_steps_per_epoch,
           stop_after_epochs) -> str:
    manifest = json.loads((processed_dir / "manifest.json").read_text())
    splits = json.loads((processed_dir / "splits.json").read_text())
    tiles = {t["id"]: t for t in manifest["tiles"]}
    split = splits[cfg["protocol"]]

    set_seed(cfg["seed"])
    model = build_research_model(cfg["model"], **cfg.get("model_kwargs", {})).to(device)
    base_trainable = {n: p.requires_grad for n, p in model.named_parameters()}
    stages = resolve_stages(cfg)
    use_amp = cfg.get("amp", True) and device.type == "cuda"
    has_alignment = hasattr(model, "estimate_flow") and getattr(model, "aligner", None) is not None
    ssl_cfg = cfg.get("ssl") if (has_alignment and cfg.get("ssl", {}).get("enabled", True)) else None

    state = {"stage": 0, "epoch": 0, "best_score": -1.0, "epoch_seconds": [], "stage_initialized": False}
    last_path = ckpt_dir / "last.pt"
    resume = torch.load(last_path, map_location="cpu", weights_only=False) if last_path.exists() else None
    if resume is not None:
        if resume["config_hash"] != chash:
            raise RuntimeError(
                f"Checkpoint config {resume['config_hash']} != current {chash}; refusing to mix settings. "
                "Delete the run folder to restart."
            )
        load_model_state(model, resume["model"])
        state = resume["state"]
        logger.info(f"Resuming {cfg['id']} at stage {state['stage']} epoch {state['epoch']}")

    if state.get("training_done"):
        return _final_evaluation(cfg, model, processed_dir, run_dir, ckpt_dir, splits, tiles, deadline, device)

    val_loader = DataLoader(
        SN8CenterCropDataset(processed_dir, split["val"], cfg["crop_size"], cfg.get("register_post", False)),
        batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg.get("num_workers", 2),
    )
    train_ds = SN8CropDataset(
        processed_dir, split["train"], cfg["crop_size"],
        aug=cfg.get("aug"), ssl=ssl_cfg, seed=cfg["seed"], register_post=cfg.get("register_post", False),
    )
    weights = tile_weights(tiles, split["train"], cfg.get("flood_oversample", 2.0))
    samples_per_epoch = int(cfg.get("samples_per_tile", 2) * len(split["train"]))
    grad_accum = cfg.get("grad_accum", 1)  # micro-batches per optimizer step (effective batch = batch_size × grad_accum)
    steps_per_epoch = samples_per_epoch // (cfg["batch_size"] * grad_accum)
    if max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, max_steps_per_epoch)

    log_path = run_dir / "train_log.csv"
    epochs_this_call = 0

    while state["stage"] < len(stages):
        stage = stages[state["stage"]]
        set_trainable(model, base_trainable, stage.get("param_prefixes"))
        if not state["stage_initialized"]:
            if stage.get("init"):
                getattr(model, stage["init"])()
                logger.info(f"Stage '{stage['name']}': ran {stage['init']}()")
            state["stage_initialized"] = True

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
        total_steps = stage["epochs"] * steps_per_epoch
        scheduler = cosine_with_warmup(optimizer, total_steps, cfg.get("warmup_epochs", 1) * steps_per_epoch)
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        ema = EMA(model, cfg.get("ema_decay", 0.999))
        criterion = MultiTaskLoss(
            stage_loss_weights(cfg, stage),
            ssl_max_px=(ssl_cfg or {}).get("max_px", 32),
            object_masked_flood=cfg.get("loss", {}).get("object_masked_flood", True),
        )
        heads = SELECTION_HEADS["joint" if stage["name"] == "joint" else stage["name"]]

        if resume is not None and resume["state"]["stage"] == state["stage"] and "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])
            scheduler.load_state_dict(resume["scheduler"])
            scaler.load_state_dict(resume["scaler"])
            ema.load_state_dict(resume["ema"])
            restore_rng_state(resume["rng"])
        resume = None

        while state["epoch"] < stage["epochs"]:
            typical = float(np.median(state["epoch_seconds"])) if state["epoch_seconds"] else 0.0
            if _time_left(deadline) < 1.3 * typical + cfg.get("deadline_margin_s", 300):
                _save_last(last_path, model, ema, optimizer, scheduler, scaler, state, chash)
                logger.info(f"Pausing before epoch {state['epoch']}: not enough session time left")
                return "paused"

            t0 = time.time()
            train_ds.set_epoch(state["stage"] * 1000 + state["epoch"])
            draws = WeightedRandomSampler(
                weights, samples_per_epoch, replacement=True,
                generator=torch.Generator().manual_seed(cfg["seed"] * 100003 + state["stage"] * 1000 + state["epoch"]),
            )
            sampler = [pos * len(train_ds) + int(t) for pos, t in enumerate(draws)]
            loader = DataLoader(
                train_ds, batch_size=cfg["batch_size"], sampler=sampler, drop_last=True,
                num_workers=cfg.get("num_workers", 2), pin_memory=device.type == "cuda",
            )
            train_metrics = _train_epoch(
                model, loader, criterion, optimizer, scheduler, scaler, ema, device, use_amp,
                cfg.get("grad_clip", 1.0), steps_per_epoch, has_alignment and ssl_cfg is not None, grad_accum,
            )

            replaced = ema.apply_to(model)
            val_metrics = validate(model, val_loader, device, cfg.get("amp", True), heads)
            improved = val_metrics["val_score"] > state["best_score"]
            if improved:
                state["best_score"] = val_metrics["val_score"]
                atomic_save({"model": model_state(model), "config_hash": chash,
                             "stage": state["stage"], "epoch": state["epoch"], "metrics": val_metrics},
                            ckpt_dir / f"best_stage{state['stage']}.pt")
            model.load_state_dict({**model.state_dict(), **replaced})

            seconds = time.time() - t0
            state["epoch_seconds"] = (state["epoch_seconds"] + [seconds])[-5:]
            row = {"stage": stage["name"], "epoch": state["epoch"], "lr": optimizer.param_groups[0]["lr"],
                   **train_metrics, **val_metrics, "best": improved, "seconds": round(seconds, 1)}
            _append_log(log_path, row)
            logger.info(
                f"[{cfg['id']}] {stage['name']} epoch {state['epoch'] + 1}/{stage['epochs']} "
                f"loss {train_metrics['loss']:.4f} val_score {val_metrics['val_score']:.4f}"
                f"{' *' if improved else ''} ({seconds:.0f}s)"
            )
            state["epoch"] += 1
            _save_last(last_path, model, ema, optimizer, scheduler, scaler, state, chash)
            epochs_this_call += 1
            if stop_after_epochs and epochs_this_call >= stop_after_epochs and state["epoch"] < stage["epochs"]:
                logger.info(f"Pausing after {epochs_this_call} epoch(s) as requested")
                return "paused"

        # Next stage starts from this stage's best (EMA) weights
        best = torch.load(ckpt_dir / f"best_stage{state['stage']}.pt", map_location="cpu", weights_only=False)
        load_model_state(model, best["model"])
        state.update({"stage": state["stage"] + 1, "epoch": 0, "best_score": -1.0, "stage_initialized": False})
        _save_last(last_path, model, None, None, None, None, state, chash)

    atomic_save({"model": model_state(model), "config_hash": chash}, ckpt_dir / "best.pt")
    state["training_done"] = True
    _save_last(last_path, model, None, None, None, None, state, chash)
    return _final_evaluation(cfg, model, processed_dir, run_dir, ckpt_dir, splits, tiles, deadline, device)


def _train_epoch(model, loader, criterion, optimizer, scheduler, scaler, ema, device, use_amp,
                 grad_clip, max_steps, use_ssl, grad_accum=1) -> dict:
    """max_steps counts optimizer steps; each takes grad_accum micro-batches."""
    train_mode(model)
    sums, n, steps = {}, 0, 0
    optimizer.zero_grad(set_to_none=True)
    for micro, batch in enumerate(loader):
        if steps >= max_steps:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(batch["pre"], batch["post"])
            ssl_flows = None
            if use_ssl and batch["ssl"].any():
                idx = batch["ssl"] > 0
                ssl_flows = model.estimate_flow(batch["ssl_a"][idx], batch["ssl_b"][idx])
            losses = criterion(outputs, batch, ssl_flows)
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"Non-finite loss at step {steps}: { {k: float(v) for k, v in losses.items()} }")
        scaler.scale(losses["total"] / grad_accum).backward()
        if (micro + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            optimizer.zero_grad(set_to_none=True)
            steps += 1

        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        n += 1
    return {("loss" if k == "total" else f"loss_{k}"): v / max(n, 1) for k, v in sums.items()}


def _save_last(path, model, ema, optimizer, scheduler, scaler, state, chash):
    ckpt = {"model": model_state(model), "state": state, "config_hash": chash, "rng": rng_state()}
    if optimizer is not None:
        ckpt.update({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "scaler": scaler.state_dict(), "ema": ema.state_dict()})
    atomic_save(ckpt, path)


def _append_log(path: Path, row: dict):
    exists = path.exists()
    fields = list(row)
    if exists:
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        fields = header + [k for k in row if k not in header]
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})


def _final_evaluation(cfg, model, processed_dir, run_dir, ckpt_dir, splits, tiles, deadline, device) -> str:
    from training.evaluate_run import final_evaluation

    best = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
    load_model_state(model, best["model"])
    model.to(device).eval()
    state = final_evaluation(cfg, model, processed_dir, run_dir, splits, tiles, deadline, device)
    if state == "completed":
        finalize_checkpoints(ckpt_dir)
    return state


def finalize_checkpoints(ckpt_dir: Path):
    """
    After a run completes, keep only best.pt (fp16 weights) so dozens of runs
    fit in a Kaggle session's output limit.
    """
    best = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
    best["model"] = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in best["model"].items()}
    best["dtype"] = "float16"
    atomic_save(best, ckpt_dir / "best.pt")
    for p in ckpt_dir.glob("*.pt"):
        if p.name != "best.pt":
            p.unlink()
