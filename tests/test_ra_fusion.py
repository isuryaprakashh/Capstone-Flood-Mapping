"""
Tests for the RA-Fusion model, its alignment module and the v0 adapters.
"""

import pytest
import torch

from models.alignment import CoarseToFineAligner, local_correlation, warp
from models.legacy import LegacyAdapter
from models.ra_fusion import FLOOD_HEADS, FOUNDATION_HEADS, RAFusionNet

OUTPUT_KEYS = set(FOUNDATION_HEADS) | set(FLOOD_HEADS) | {"flood_image"}


def _inputs(size=64, batch=2):
    torch.manual_seed(0)
    return torch.rand(batch, 3, size, size), torch.rand(batch, 3, size, size)


def test_local_correlation_peaks_at_true_displacement():
    torch.manual_seed(0)
    a = torch.randn(1, 8, 12, 12)
    b = torch.roll(a, shifts=(1, 2), dims=(2, 3))  # content moved down 1, right 2
    corr = local_correlation(a, b, radius=3)
    k = corr[0, :, 6, 6].argmax().item()
    dy, dx = divmod(k, 7)
    assert (dy - 3, dx - 3) == (1, 2)


def test_warp_inverts_a_known_shift():
    a = torch.zeros(1, 1, 16, 16)
    a[0, 0, 8, 5] = 1.0
    b = torch.roll(a, shifts=3, dims=3)
    flow = torch.zeros(1, 2, 16, 16)
    flow[:, 0] = 3.0
    assert torch.allclose(warp(b, flow), a, atol=1e-5)


def test_aligner_starts_at_identity_and_covers_all_levels():
    channels = [8, 16, 32, 32, 32]
    aligner = CoarseToFineAligner(channels)
    feats = [torch.randn(1, c, 64 // s, 64 // s) for c, s in zip(channels, (2, 4, 8, 16, 32))]
    aligned, flows, conf = aligner(feats, feats)
    assert set(flows) == {2, 4, 8, 16, 32}
    assert all(torch.count_nonzero(f) == 0 for f in flows.values())  # zero-initialised flow head
    assert conf.shape == (1, 1, 8, 8)
    assert all(torch.allclose(x, y, atol=1e-5) for x, y in zip(aligned, feats))


@pytest.mark.parametrize("align,fusion", [(True, "gated"), (False, "gated"), (True, "concat")])
def test_ra_fusion_outputs_and_gradients(align, fusion):
    model = RAFusionNet(backbone="resnet18", pretrained=False, align=align, fusion=fusion)
    pre, post = _inputs()
    out = model(pre, post)

    assert OUTPUT_KEYS <= set(out)
    for k in set(FOUNDATION_HEADS) | set(FLOOD_HEADS):
        assert out[k].shape == (2, 1, 64, 64)
    assert out["flood_image"].shape == (2, 1)
    assert ("flows" in out) == align

    loss = sum(out[k].mean() for k in OUTPUT_KEYS)
    if align:
        loss = loss + sum(f.abs().mean() for f in out["flows"].values()) + out["align_conf"].mean()
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"parameters without gradients: {missing[:5]}"


def test_estimate_flow_shapes_and_requires_alignment():
    model = RAFusionNet(backbone="resnet18", pretrained=False)
    pre, post = _inputs()
    flows = model.estimate_flow(pre, post)
    assert flows[16].shape == (2, 2, 4, 4) and flows[2].shape == (2, 2, 32, 32)

    no_align = RAFusionNet(backbone="resnet18", pretrained=False, align=False)
    with pytest.raises(RuntimeError):
        no_align.estimate_flow(pre, post)


def test_dinov3_backbone_is_frozen():
    model = RAFusionNet(backbone="dinov3_sat", pretrained=False)
    vit_trainable = [n for n, p in model.backbone.vit.named_parameters() if p.requires_grad]
    assert not vit_trainable
    model.train()
    assert not model.backbone.vit.training  # frozen ViT stays in eval mode
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 10e6 < trainable < 40e6

    pre, post = _inputs(size=64, batch=1)
    out = model(pre, post)
    assert out["building"].shape == (1, 1, 64, 64)


@pytest.mark.parametrize("kind", ["unet_v0", "siamese_v0"])
def test_legacy_adapters(kind):
    model = LegacyAdapter(kind=kind, pretrained=False)
    pre, post = _inputs()
    out = model(pre, post)
    assert OUTPUT_KEYS <= set(out)
    assert out["road"].shape == (2, 1, 64, 64) and out["flood_image"].shape == (2, 1)
