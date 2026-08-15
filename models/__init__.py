"""
Model factory for SpaceNet 8 Flood Mapping.

Provides a unified interface to create model instances by name.
"""

from models.unet_baseline import UNetBaseline
from models.siamese_fusion import SiameseFusionNet


MODEL_REGISTRY = {
    "unet": UNetBaseline,
    "unet_baseline": UNetBaseline,
    "fusion": SiameseFusionNet,
    "siamese_fusion": SiameseFusionNet,
    "siamese": SiameseFusionNet,
}


def build_model(name: str, **kwargs) -> "torch.nn.Module":
    """
    Build a model by name.

    Args:
        name: Model name (one of: 'unet', 'fusion', 'siamese_fusion').
        **kwargs: Model constructor arguments.

    Returns:
        Instantiated model.

    Raises:
        ValueError: If model name is not recognized.
    """
    name_lower = name.lower().replace("-", "_")

    if name_lower not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model '{name}'. Available: {available}"
        )

    model_cls = MODEL_REGISTRY[name_lower]
    return model_cls(**kwargs)


def count_parameters(model) -> dict:
    """
    Count model parameters.

    Returns:
        Dictionary with total, trainable, and frozen parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "total_millions": round(total / 1e6, 2),
    }


__all__ = [
    "UNetBaseline",
    "SiameseFusionNet",
    "build_model",
    "count_parameters",
    "MODEL_REGISTRY",
]
