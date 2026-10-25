"""
Published baselines for SpaceNet 8 flood mapping.

All baselines share one interface: ``forward(pre, post)`` takes raw RGB
tensors in [0, 1] (B, 3, H, W), H and W divisible by 32, and returns a
dict of full-resolution logits keyed by OUTPUT_KEYS.

  - SN8Baseline: SpaceNet 8 official baseline (Hänsch et al., CVPRW 2022).
  - SiameseUNet: Siamese U-Net of the SpaceNet 8 5th-place solution
    (motokimura, 2022).
  - SNUNetCD: SNUNet-CD/32 with ECAM (Fang et al., IEEE GRSL 2021).
"""

import torch.nn as nn

from models.baselines.common import FLOOD_KEYS, FOUNDATION_KEYS, OUTPUT_KEYS
from models.baselines.siamese_unet import SiameseUNet
from models.baselines.sn8_baseline import SN8Baseline
from models.baselines.snunet import SNUNetCD


BASELINE_REGISTRY = {
    "sn8_baseline": SN8Baseline,
    "siamese_unet": SiameseUNet,
    "snunet": SNUNetCD,
}


def build_baseline(name: str, **kwargs) -> nn.Module:
    """
    Build a baseline by name.

    Args:
        name: One of 'sn8_baseline', 'siamese_unet', 'snunet'.
        **kwargs: Model constructor arguments.

    Returns:
        Instantiated model.

    Raises:
        ValueError: If the name is not recognised.
    """
    key = name.lower().replace("-", "_")
    if key not in BASELINE_REGISTRY:
        available = ", ".join(sorted(BASELINE_REGISTRY))
        raise ValueError(f"Unknown baseline '{name}'. Available: {available}")
    return BASELINE_REGISTRY[key](**kwargs)


__all__ = [
    "SN8Baseline",
    "SiameseUNet",
    "SNUNetCD",
    "BASELINE_REGISTRY",
    "build_baseline",
    "OUTPUT_KEYS",
    "FOUNDATION_KEYS",
    "FLOOD_KEYS",
]
