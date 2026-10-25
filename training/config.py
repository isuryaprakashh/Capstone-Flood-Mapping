"""
Experiment Configuration.

Runs are defined in experiments/runs.yaml: a `defaults` block shared by every
run and a `runs` list whose entries override any default. A run's resolved
configuration is hashed; resuming a checkpoint with a different configuration
is refused so no run silently mixes settings.
"""

import copy
import hashlib
import json
from pathlib import Path
from typing import Union

import yaml

DEFAULT_RUNS_FILE = Path(__file__).resolve().parent.parent / "experiments" / "runs.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_runs(path: Union[str, Path] = DEFAULT_RUNS_FILE) -> dict[str, dict]:
    """
    Load and resolve every run in a runs file.

    Returns:
        Mapping run id → resolved configuration (defaults merged in).
    """
    spec = yaml.safe_load(Path(path).read_text())
    defaults = spec.get("defaults", {})
    runs = {}
    for entry in spec.get("runs", []):
        cfg = deep_merge(defaults, entry)
        if cfg["id"] in runs:
            raise ValueError(f"Duplicate run id '{cfg['id']}'")
        runs[cfg["id"]] = cfg
    return runs


def config_hash(cfg: dict) -> str:
    """
    Stable hash of the settings that affect training results.

    Bookkeeping fields (queue, priority, notes) are excluded so a run can be
    moved between queues without invalidating its checkpoint.
    """
    relevant = {k: v for k, v in cfg.items() if k not in ("queue", "priority", "notes")}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]
