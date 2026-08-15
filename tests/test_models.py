"""
Unit tests for model architectures: U-Net baseline and Siamese Cross-Attention.
"""

import pytest
import torch
from models import build_model, count_parameters
from models.unet_baseline import UNetBaseline
from models.siamese_fusion import SiameseFusionNet
from losses.combined_loss import CombinedLoss, DiceLoss, FocalLoss


def test_unet_baseline_forward(sample_batch):
    """Test baseline U-Net forward pass and output shape."""
    model = UNetBaseline(in_channels=6, num_classes=5, pretrained=False)
    pre = sample_batch["pre_image"]
    post = sample_batch["post_image"]

    logits = model(pre, post)
    assert logits.shape == (2, 5, 128, 128)
    assert not torch.isnan(logits).any()


def test_unet_baseline_predict(sample_batch):
    """Test U-Net predict method generates binary masks."""
    model = UNetBaseline(in_channels=6, num_classes=5, pretrained=False)
    pre = sample_batch["pre_image"]
    post = sample_batch["post_image"]

    preds = model.predict(pre, post, threshold=0.5)
    assert preds.shape == (2, 5, 128, 128)
    assert set(preds.unique().tolist()).issubset({0.0, 1.0})


def test_siamese_fusion_forward(sample_batch):
    """Test Siamese Fusion forward pass and output shape."""
    model = SiameseFusionNet(in_channels=3, num_classes=5, pretrained=False)
    pre = sample_batch["pre_image"]
    post = sample_batch["post_image"]

    logits = model(pre, post)
    assert logits.shape == (2, 5, 128, 128)
    assert not torch.isnan(logits).any()


def test_siamese_fusion_predict(sample_batch):
    """Test Siamese Fusion predict method generates binary masks."""
    model = SiameseFusionNet(in_channels=3, num_classes=5, pretrained=False)
    pre = sample_batch["pre_image"]
    post = sample_batch["post_image"]

    preds = model.predict(pre, post, threshold=0.5)
    assert preds.shape == (2, 5, 128, 128)
    assert set(preds.unique().tolist()).issubset({0.0, 1.0})


def test_model_factory():
    """Test model registry and factory function."""
    unet = build_model("unet", in_channels=6, num_classes=5, pretrained=False)
    assert isinstance(unet, UNetBaseline)

    fusion = build_model("siamese_fusion", in_channels=3, num_classes=5, pretrained=False)
    assert isinstance(fusion, SiameseFusionNet)

    with pytest.raises(ValueError):
        build_model("unknown_arch")


def test_parameter_counter():
    """Test parameter counter utility."""
    model = UNetBaseline(in_channels=6, num_classes=5, pretrained=False)
    stats = count_parameters(model)
    assert "total" in stats
    assert "trainable" in stats
    assert stats["total"] > 0


def test_combined_loss(sample_batch):
    """Test Dice + Focal + BCE combined loss calculation and gradient flow."""
    criterion = CombinedLoss(dice_weight=1.0, focal_weight=1.0, bce_weight=0.5)
    logits = torch.randn(2, 5, 64, 64, requires_grad=True)
    targets = sample_batch["mask"][:, :, :64, :64]

    loss_dict = criterion(logits, targets)
    assert "total" in loss_dict
    assert "dice" in loss_dict
    assert "focal" in loss_dict
    assert "bce" in loss_dict

    loss = loss_dict["total"]
    loss.backward()
    assert logits.grad is not None
