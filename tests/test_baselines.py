"""
Unit tests for the published baselines: SN8 baseline, Siamese U-Net
(SpaceNet 8 5th place) and SNUNet-CD.
"""

import pytest
import torch

from models.baselines import (
    BASELINE_REGISTRY,
    OUTPUT_KEYS,
    SiameseUNet,
    SN8Baseline,
    SNUNetCD,
    build_baseline,
)
from models.baselines.common import IMAGENET_MEAN, IMAGENET_STD

SMALL_BUILDERS = {
    "sn8_resnet18": lambda: SN8Baseline(encoder_name="resnet18", pretrained=False),
    "siamese_effb0": lambda: SiameseUNet(encoder_name="efficientnet_b0", pretrained=False),
    "siamese_resnet18": lambda: SiameseUNet(encoder_name="resnet18", pretrained=False),
    "snunet_c8": lambda: SNUNetCD(base_channels=8),
}


def _count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _inputs(batch: int = 2, size: int = 64, dtype: torch.dtype = torch.float32):
    torch.manual_seed(0)
    pre = torch.rand(batch, 3, size, size, dtype=dtype)
    post = torch.rand(batch, 3, size, size, dtype=dtype)
    return pre, post


def _check_contract(out: dict, batch: int, size: int) -> None:
    assert set(out) == set(OUTPUT_KEYS)
    for key, value in out.items():
        expected = (batch, 1) if key == "flood_image" else (batch, 1, size, size)
        assert tuple(value.shape) == expected, key
        assert torch.isfinite(value).all(), key


@pytest.mark.parametrize("name", list(SMALL_BUILDERS))
def test_forward_contract(name):
    """Raw [0, 1] inputs give every output key at full resolution."""
    model = SMALL_BUILDERS[name]()
    pre, post = _inputs()
    assert pre.min() >= 0 and pre.max() <= 1
    _check_contract(model(pre, post), batch=2, size=64)


@pytest.mark.parametrize("name", list(SMALL_BUILDERS))
def test_backward_reaches_all_parameters(name):
    """Sum of output means back-propagates into every parameter."""
    model = SMALL_BUILDERS[name]()
    pre, post = _inputs()
    out = model(pre, post)
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing[:5]
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("name", ["sn8_resnet18", "snunet_c8"])
def test_eval_mode_and_autocast(name):
    """Eval-mode forward works under (CPU bf16) autocast."""
    model = SMALL_BUILDERS[name]().eval()
    pre, post = _inputs()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(pre, post)
    _check_contract({k: v.float() for k, v in out.items()}, batch=2, size=64)


def test_small_parameter_counts():
    """Report parameter counts of the light test configurations."""
    for name, builder in SMALL_BUILDERS.items():
        print(f"{name}: {_count(builder()) / 1e6:.2f}M parameters")


def test_full_size_parameter_counts():
    """Full-size configs are in the expected parameter range."""
    sn8 = SN8Baseline(pretrained=False)
    siamese = SiameseUNet(pretrained=False)
    snunet = SNUNetCD()

    snunet_aux = (
        _count(snunet.foundation_decoder)
        + _count(snunet.foundation_heads)
        + _count(snunet.flood_image)
    )
    counts = {
        "sn8_baseline (resnet34)": _count(sn8),
        "  sn8 foundation": _count(sn8.foundation),
        "  sn8 flood": _count(sn8.flood),
        "siamese_unet (efficientnet_b5)": _count(siamese),
        "  siamese foundation": _count(siamese.foundation),
        "  siamese flood": _count(siamese.flood),
        "snunet (c32)": _count(snunet),
        "  SNUNet-CD/32 core": _count(snunet) - snunet_aux,
        "  snunet auxiliary paths": snunet_aux,
    }
    for name, value in counts.items():
        print(f"{name}: {value:,} ({value / 1e6:.2f}M)")

    assert 45e6 < _count(sn8) < 53e6
    assert 55e6 < _count(siamese) < 66e6
    assert 12.5e6 < _count(snunet) < 14.5e6
    assert 11.9e6 < _count(snunet) - snunet_aux < 12.2e6  # paper: 12.03M


@pytest.mark.parametrize("name", ["sn8_resnet18", "siamese_effb0"])
def test_stage_parameters_partition(name):
    """Foundation and flood stages partition the parameters."""
    model = SMALL_BUILDERS[name]()
    foundation = {id(p) for p in model.stage_parameters("foundation")}
    flood = {id(p) for p in model.stage_parameters("flood")}
    everything = {id(p) for p in model.parameters()}

    assert foundation and flood
    assert not foundation & flood
    assert foundation | flood == everything

    with pytest.raises(ValueError):
        model.stage_parameters("decoder")


def test_init_flood_from_foundation_copies_weights():
    """Flood U-Net receives an independent copy of the foundation U-Net."""
    model = SMALL_BUILDERS["siamese_resnet18"]()
    src = model.foundation.unet.state_dict()
    dst = model.flood.unet.state_dict()
    assert src.keys() == dst.keys()
    assert any(
        not torch.equal(src[k], dst[k]) for k in src if src[k].is_floating_point()
    )

    model.init_flood_from_foundation()
    dst = model.flood.unet.state_dict()
    for key in src:
        assert torch.equal(src[key], dst[key]), key
        assert src[key].data_ptr() != dst[key].data_ptr(), key

    with torch.no_grad():
        next(model.foundation.unet.parameters()).add_(1.0)
    assert not torch.equal(
        next(model.foundation.unet.parameters()), next(model.flood.unet.parameters())
    )


@pytest.mark.parametrize("name", ["sn8_resnet18", "siamese_effb0", "snunet_c8"])
def test_normalization_buffers(name):
    """Models normalise internally and their stats move with .to()."""
    model = SMALL_BUILDERS[name]()
    buffers = dict(model.named_buffers())
    assert "mean" in buffers and "std" in buffers
    assert "mean" not in model.state_dict()

    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x = torch.rand(1, 3, 8, 8)
    assert torch.allclose(model.normalize(x), (x - mean) / std)

    model = model.to(torch.float64)
    assert model.mean.dtype == torch.float64 and model.std.dtype == torch.float64
    pre, post = _inputs(batch=1, size=32, dtype=torch.float64)
    out = model.eval()(pre, post)
    assert out["building"].dtype == torch.float64

    if torch.cuda.is_available():
        model = model.to("cuda")
        assert model.mean.device.type == "cuda" and model.std.device.type == "cuda"


def test_build_baseline_registry():
    """Factory builds every registered baseline and rejects unknown names."""
    assert isinstance(build_baseline("sn8-baseline", encoder_name="resnet18", pretrained=False), SN8Baseline)
    assert isinstance(build_baseline("siamese_unet", encoder_name="resnet18", pretrained=False), SiameseUNet)
    assert isinstance(build_baseline("snunet", base_channels=8), SNUNetCD)
    assert set(BASELINE_REGISTRY) == {"sn8_baseline", "siamese_unet", "snunet"}
    with pytest.raises(ValueError):
        build_baseline("unknown")
