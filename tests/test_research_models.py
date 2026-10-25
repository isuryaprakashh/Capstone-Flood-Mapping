"""
Every research model builds through the factory and honours the shared
training interface (same inputs, same output keys and shapes).
"""

import pytest
import torch

from models import RESEARCH_MODELS, build_research_model

SMALL_CONFIGS = {
    "ra_fusion": {"backbone": "resnet18", "pretrained": False},
    "sn8_baseline": {"encoder_name": "resnet18", "pretrained": False},
    "siamese_unet": {"encoder_name": "resnet18", "pretrained": False},
    "snunet": {"base_channels": 8, "pretrained": False},  # factory drops 'pretrained'
    "unet_v0": {"pretrained": False},
    "siamese_v0": {"pretrained": False},
}
PIXEL_KEYS = ("building", "building_border", "road", "junction", "flooded_building", "flooded_road")


def test_small_configs_cover_every_research_model():
    assert set(SMALL_CONFIGS) == set(RESEARCH_MODELS)


@pytest.mark.parametrize("name", RESEARCH_MODELS)
def test_factory_builds_model_with_shared_interface(name):
    torch.manual_seed(0)
    model = build_research_model(name, **SMALL_CONFIGS[name])
    pre, post = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)
    out = model(pre, post)
    for k in PIXEL_KEYS:
        assert out[k].shape == (2, 1, 64, 64), k
        assert torch.isfinite(out[k]).all(), k
    assert out["flood_image"].shape == (2, 1)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        build_research_model("does_not_exist")
