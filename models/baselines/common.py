"""
Shared building blocks for the SpaceNet 8 flood-mapping baselines.

Every baseline follows the same interface: ``forward(pre, post)`` takes raw
RGB tensors in [0, 1], normalises them internally with the encoder's
statistics and returns a dict of full-resolution logits (see OUTPUT_KEYS).

This module provides the base classes implementing that contract, a timm
feature-pyramid encoder factory, a U-Net decoder in the style of
segmentation_models_pytorch (Iakubovskii, 2019; Ronneberger et al.,
MICCAI 2015), the per-output prediction heads and the image-level
"any flood" pooling head.
"""

from collections.abc import Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

FOUNDATION_KEYS = ("building", "building_border", "road", "junction")
FLOOD_KEYS = ("flooded_building", "flooded_road")
OUTPUT_KEYS = FOUNDATION_KEYS + FLOOD_KEYS + ("flood_image",)

PYRAMID_REDUCTIONS = [2, 4, 8, 16, 32]


class BaselineModel(nn.Module):
    """
    Base class holding the input normalisation shared by all baselines.

    Registers ``mean`` and ``std`` buffers of shape (1, 3, 1, 1) so that
    they follow the model across ``.to(device/dtype)``. They are
    non-persistent: they are derived from the encoder and are not stored
    in checkpoints.

    Args:
        mean: Per-channel mean of the encoder's pretraining data.
        std: Per-channel standard deviation of the encoder's pretraining data.
    """

    def __init__(
        self,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ):
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )

    def set_normalization(self, mean: Sequence[float], std: Sequence[float]) -> None:
        """Overwrite the normalisation statistics in place."""
        with torch.no_grad():
            self.mean.copy_(torch.tensor(mean).view_as(self.mean))
            self.std.copy_(torch.tensor(std).view_as(self.std))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalise a raw RGB batch.

        Args:
            x: Image tensor (B, 3, H, W) with values in [0, 1].

        Returns:
            Normalised tensor (B, 3, H, W).
        """
        return (x - self.mean) / self.std


class TwoStageModel(BaselineModel):
    """
    Baseline made of a separate foundation network and flood network.

    Subclasses must define ``self.foundation`` and ``self.flood``; every
    trainable parameter has to live in one of them so that the two
    networks can be trained in separate stages.
    """

    STAGES = ("foundation", "flood")

    def stage_parameters(self, stage: str) -> list[nn.Parameter]:
        """
        Return the parameters trained in a given stage.

        Args:
            stage: 'foundation' or 'flood'.

        Returns:
            List of parameters of the corresponding sub-network.

        Raises:
            ValueError: If the stage name is not recognised.
        """
        if stage not in self.STAGES:
            raise ValueError(f"Unknown stage '{stage}'. Expected one of {self.STAGES}.")
        return list(getattr(self, stage).parameters())


def encoder_stats(encoder: nn.Module) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """
    Read the normalisation statistics of a timm encoder.

    Args:
        encoder: Model created with ``timm.create_model``.

    Returns:
        (mean, std) tuples, falling back to ImageNet statistics.
    """
    cfg = getattr(encoder, "pretrained_cfg", None) or {}
    return tuple(cfg.get("mean", IMAGENET_MEAN)), tuple(cfg.get("std", IMAGENET_STD))


def create_encoder(name: str, pretrained: bool = True) -> nn.Module:
    """
    Create a timm encoder returning a 5-level feature pyramid.

    Args:
        name: timm model name, optionally with a pretrained tag
              (e.g. 'resnet34.tv_in1k').
        pretrained: Load pretrained weights.

    Returns:
        timm features_only model producing features at strides 2..32.

    Raises:
        ValueError: If the encoder does not expose strides [2, 4, 8, 16, 32].
    """
    encoder = timm.create_model(
        name, features_only=True, pretrained=pretrained, out_indices=(0, 1, 2, 3, 4)
    )
    reductions = list(encoder.feature_info.reduction())
    if reductions != PYRAMID_REDUCTIONS:
        raise ValueError(
            f"Encoder '{name}' has feature strides {reductions}; "
            f"expected {PYRAMID_REDUCTIONS}."
        )
    return encoder


def kaiming_init(module: nn.Module) -> None:
    """Kaiming-normal (fan_out, ReLU) init for convs; unit/zero init for BN."""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class ConvBNReLU(nn.Sequential):
    """3x3 convolution → (BatchNorm) → ReLU."""

    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """
    U-Net decoder block: 2x nearest upsampling → skip concat → 2x ConvBNReLU.

    Args:
        in_channels: Channels of the incoming (deeper) features.
        skip_channels: Channels of the skip connection (0 for none).
        out_channels: Output channels.
        use_batchnorm: Use BatchNorm after each convolution.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels + skip_channels, out_channels, use_batchnorm)
        self.conv2 = ConvBNReLU(out_channels, out_channels, use_batchnorm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class UNetDecoder(nn.Module):
    """
    U-Net decoder over a feature pyramid (segmentation_models_pytorch style).

    Args:
        encoder_channels: Channels of the pyramid levels, shallow to deep;
            each level has half the resolution of the previous one.
        decoder_channels: Output channels of each decoder block, deep to
            shallow. Blocks beyond the available skip connections upsample
            without a skip (e.g. the stride-2 → stride-1 block of a timm
            encoder).
        use_batchnorm: Use BatchNorm in the decoder blocks.

    Raises:
        ValueError: If there are fewer decoder blocks than skip connections.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        use_batchnorm: bool = True,
    ):
        super().__init__()
        encoder_channels = list(encoder_channels)
        decoder_channels = list(decoder_channels)
        n_skips = len(encoder_channels) - 1
        if len(decoder_channels) < n_skips:
            raise ValueError(
                f"Need at least {n_skips} decoder blocks, got {len(decoder_channels)}."
            )

        skip_channels = encoder_channels[-2::-1] + [0] * (len(decoder_channels) - n_skips)
        in_channels = [encoder_channels[-1]] + decoder_channels[:-1]
        self.blocks = nn.ModuleList(
            DecoderBlock(i, s, o, use_batchnorm)
            for i, s, o in zip(in_channels, skip_channels, decoder_channels)
        )
        self.out_channels = decoder_channels[-1]

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Decode a feature pyramid.

        Args:
            features: Pyramid features, shallow to deep.

        Returns:
            Decoded features with ``out_channels`` channels.
        """
        skips = list(features[-2::-1])
        x = features[-1]
        for i, block in enumerate(self.blocks):
            x = block(x, skips[i] if i < len(skips) else None)
        return x


class TimmUNet(nn.Module):
    """
    U-Net with a timm encoder; returns full-resolution decoder features.

    Args:
        encoder_name: timm model name.
        pretrained: Load pretrained encoder weights.
        decoder_channels: Five decoder widths, deep to shallow; the last
            block brings the features back to input resolution.
    """

    def __init__(
        self,
        encoder_name: str,
        pretrained: bool = True,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
    ):
        super().__init__()
        if len(decoder_channels) != len(PYRAMID_REDUCTIONS):
            raise ValueError("TimmUNet needs exactly five decoder widths.")
        self.encoder = create_encoder(encoder_name, pretrained)
        self.decoder = UNetDecoder(self.encoder.feature_info.channels(), decoder_channels)
        kaiming_init(self.decoder)
        self.out_channels = self.decoder.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def conv_head(in_channels: int, kernel_size: int = 1) -> nn.Conv2d:
    """Single-logit prediction head (``kernel_size`` x ``kernel_size`` conv)."""
    return nn.Conv2d(in_channels, 1, kernel_size, padding=kernel_size // 2)


def build_heads(
    in_channels: int, names: Sequence[str], kernel_size: int = 1
) -> nn.ModuleDict:
    """
    Build one single-logit conv head per output name.

    Args:
        in_channels: Channels of the shared features.
        names: Output keys.
        kernel_size: Convolution kernel size.

    Returns:
        ModuleDict mapping each name to its head.
    """
    return nn.ModuleDict({name: conv_head(in_channels, kernel_size) for name in names})


def apply_heads(heads: nn.ModuleDict, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Apply every head of a ModuleDict to the same features."""
    return {name: head(x) for name, head in heads.items()}


class FloodImageHead(nn.Module):
    """
    Image-level "any flood" logit from globally pooled flood features.

    Concatenates global average and global max pooling and applies a
    linear layer. Averaging accumulates in at least float32 so that the
    full-resolution mean is safe under fp16 autocast.

    Args:
        in_channels: Channels of the flood features.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.fc = nn.Linear(2 * in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Flood features (B, C, H, W).

        Returns:
            Logits (B, 1).
        """
        acc = torch.promote_types(x.dtype, torch.float32)
        avg = x.mean(dim=(2, 3), dtype=acc)
        mx = x.amax(dim=(2, 3)).to(acc)
        pooled = torch.cat([avg, mx], dim=1).to(self.fc.weight.dtype)
        return self.fc(pooled)
