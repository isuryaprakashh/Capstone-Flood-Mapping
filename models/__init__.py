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


RESEARCH_MODELS = ("ra_fusion", "sn8_baseline", "siamese_unet", "snunet", "unet_v0", "siamese_v0")


def build_research_model(name: str, **kwargs) -> "torch.nn.Module":
    """
    Build a model on the research training interface.

    All research models take pre/post RGB in [0, 1] and return a dict of
    logits (building, building_border, road, junction, flooded_building,
    flooded_road at input resolution, and flood_image per sample).

    Args:
        name: One of RESEARCH_MODELS.
        **kwargs: Model constructor arguments (e.g. pretrained, backbone).
    """
    if name == "ra_fusion":
        from models.ra_fusion import RAFusionNet
        return RAFusionNet(**kwargs)
    if name in ("unet_v0", "siamese_v0"):
        from models.legacy import LegacyAdapter
        return LegacyAdapter(kind=name, **kwargs)
    if name == "sn8_baseline":
        from models.baselines.sn8_baseline import SN8Baseline
        return SN8Baseline(**kwargs)
    if name == "siamese_unet":
        from models.baselines.siamese_unet import SiameseUNet
        return SiameseUNet(**kwargs)
    if name == "snunet":
        from models.baselines.snunet import SNUNetCD
        kwargs.pop("pretrained", None)  # trained from scratch, as in the paper
        return SNUNetCD(**kwargs)
    raise ValueError(f"Unknown research model '{name}'. Available: {', '.join(RESEARCH_MODELS)}")


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
    "build_research_model",
    "count_parameters",
    "MODEL_REGISTRY",
    "RESEARCH_MODELS",
]
