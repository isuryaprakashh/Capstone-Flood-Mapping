"""
SpaceNet 8 official baseline.

Re-implementation of the two-network baseline of
    Hänsch, R. et al., "SpaceNet 8 - The Detection of Flooded Roads and
    Buildings", CVPR Workshops (EarthVision), 2022,
following the released baseline code (SpaceNetChallenge/SpaceNet8,
baseline/models/pytorch_zoo):

  - Foundation features network: U-Net with a ResNet34 encoder on the
    pre-event image, predicting buildings and roads.
  - Flood attribution network: a separate Siamese U-Net with a ResNet34
    encoder whose weights are shared between the pre- and post-event
    branches. The two branches' final decoder features are concatenated
    and passed through two convolutions to produce the flood outputs.

The decoder reproduces the baseline's CRESI-style U-Net: each stage is a
2x nearest upsampling followed by conv3x3-ReLU, concatenation with the
encoder skip and another conv3x3-ReLU (no BatchNorm), with widths
mirroring the encoder and a final upsampling to 32 full-resolution
channels.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn

from models.baselines.common import (
    FloodImageHead,
    TwoStageModel,
    apply_heads,
    conv_head,
    create_encoder,
    encoder_stats,
    kaiming_init,
)


def _conv_relu(in_channels: int, out_channels: int) -> nn.Sequential:
    """3x3 convolution → ReLU (ConvBottleneck of the baseline code)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )


def _up_conv_relu(in_channels: int, out_channels: int) -> nn.Sequential:
    """2x nearest upsampling → 3x3 convolution → ReLU (UnetDecoderBlock)."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class SN8Decoder(nn.Module):
    """
    Decoder of the SpaceNet 8 baseline U-Net.

    Args:
        encoder_channels: Channels of the five encoder stages, shallow to
            deep (64, 64, 128, 256, 512 for ResNet34).
    """

    def __init__(self, encoder_channels: Sequence[int]):
        super().__init__()
        filters = list(encoder_channels)
        levels = range(len(filters) - 1, 0, -1)
        self.up_blocks = nn.ModuleList(
            _up_conv_relu(filters[i], filters[i - 1]) for i in levels
        )
        self.fuse_blocks = nn.ModuleList(
            _conv_relu(2 * filters[i - 1], filters[i - 1]) for i in levels
        )
        self.last_upsample = _up_conv_relu(filters[0], filters[0] // 2)
        self.out_channels = filters[0] // 2

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: Encoder features at strides 2..32, shallow to deep.

        Returns:
            Full-resolution decoder features (B, out_channels, H, W).
        """
        x = features[-1]
        for up, fuse, skip in zip(self.up_blocks, self.fuse_blocks, features[-2::-1]):
            x = fuse(torch.cat([up(x), skip], dim=1))
        return self.last_upsample(x)


class SN8UNet(nn.Module):
    """
    Baseline U-Net trunk: timm encoder + SN8Decoder.

    Args:
        encoder_name: timm encoder name.
        pretrained: Load pretrained encoder weights.
    """

    def __init__(self, encoder_name: str = "resnet34.tv_in1k", pretrained: bool = True):
        super().__init__()
        self.encoder = create_encoder(encoder_name, pretrained)
        self.decoder = SN8Decoder(self.encoder.feature_info.channels())
        kaiming_init(self.decoder)
        self.out_channels = self.decoder.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class SN8FoundationNet(nn.Module):
    """
    Foundation features network (pre-event image only).

    Building and road use the baseline's 3x3 classifier convolutions.
    The building border and road junction heads are auxiliary 1x1 heads
    added for a uniform training interface (the baseline predicts neither).

    Args:
        encoder_name: timm encoder name.
        pretrained: Load pretrained encoder weights.
    """

    def __init__(self, encoder_name: str = "resnet34.tv_in1k", pretrained: bool = True):
        super().__init__()
        self.unet = SN8UNet(encoder_name, pretrained)
        c = self.unet.out_channels
        self.heads = nn.ModuleDict({
            "building": conv_head(c, 3),
            "road": conv_head(c, 3),
            "building_border": conv_head(c, 1),
            "junction": conv_head(c, 1),
        })

    def forward(self, pre: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            pre: Normalised pre-event image (B, 3, H, W).

        Returns:
            Dict of foundation logits, each (B, 1, H, W).
        """
        return apply_heads(self.heads, self.unet(pre))


class SN8FloodNet(nn.Module):
    """
    Flood attribution network: shared-weight Siamese U-Net.

    The two 32-channel branch outputs are concatenated and passed through
    the baseline's penultimate conv3x3 (64→32, no activation) and its 3x3
    classifier, here split into one head per flood output.

    Args:
        encoder_name: timm encoder name.
        pretrained: Load pretrained encoder weights.
    """

    def __init__(self, encoder_name: str = "resnet34.tv_in1k", pretrained: bool = True):
        super().__init__()
        self.unet = SN8UNet(encoder_name, pretrained)
        c = self.unet.out_channels
        self.penultimate_conv = nn.Conv2d(2 * c, c, 3, padding=1)
        self.heads = nn.ModuleDict({
            "flooded_building": conv_head(c, 3),
            "flooded_road": conv_head(c, 3),
        })
        kaiming_init(self.penultimate_conv)
        self.flood_image = FloodImageHead(c)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            pre: Normalised pre-event image (B, 3, H, W).
            post: Normalised post-event image (B, 3, H, W).

        Returns:
            Dict with flood logits (B, 1, H, W) and 'flood_image' (B, 1).
        """
        x = torch.cat([self.unet(pre), self.unet(post)], dim=1)
        x = self.penultimate_conv(x)
        out = apply_heads(self.heads, x)
        out["flood_image"] = self.flood_image(x)
        return out


class SN8Baseline(TwoStageModel):
    """
    SpaceNet 8 baseline: foundation U-Net + Siamese flood U-Net.

    The two networks share nothing and can be trained in two stages via
    ``stage_parameters('foundation')`` and ``stage_parameters('flood')``.
    Outputs follow the common contract; the building border, road
    junction and image-level flood heads are auxiliary heads added for a
    uniform training interface. Road speed channels of the original are
    not modelled.

    Args:
        encoder_name: timm encoder for both networks. Defaults to the
            torchvision ImageNet ResNet34 weights used by the original.
        pretrained: Load pretrained encoder weights.
    """

    def __init__(self, encoder_name: str = "resnet34.tv_in1k", pretrained: bool = True):
        super().__init__()
        self.foundation = SN8FoundationNet(encoder_name, pretrained)
        self.flood = SN8FloodNet(encoder_name, pretrained)
        self.set_normalization(*encoder_stats(self.foundation.unet.encoder))

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            pre: Pre-event RGB image (B, 3, H, W) in [0, 1].
            post: Post-event RGB image (B, 3, H, W) in [0, 1].

        Returns:
            Dict of logits: per-pixel outputs (B, 1, H, W) and
            'flood_image' (B, 1).
        """
        pre = self.normalize(pre)
        post = self.normalize(post)
        out = self.foundation(pre)
        out.update(self.flood(pre, post))
        return out
