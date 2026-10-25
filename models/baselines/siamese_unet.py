"""
Siamese U-Net baseline in the style of the SpaceNet 8 5th-place solution.

Re-implementation of the EfficientNet models of motokimura's SpaceNet 8
solution (github.com/motokimura/spacenet8_solution; SpaceNet 8 challenge,
2022), built on the segmentation_models_pytorch U-Net design:

  - Foundation network: U-Net with an EfficientNet-B5 encoder on the
    pre-event image predicting building body, building border, road and
    road junction maps through a 3x3 segmentation head.
  - Flood network: a Siamese U-Net of the same architecture. Pre- and
    post-event images go through the same U-Net (shared weights); the
    two decoder outputs are concatenated (pre first) and passed through
    a 7x7 convolution and a 1x1 convolution. The flood U-Net is
    initialised from the foundation U-Net before flood training.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn

from models.baselines.common import (
    FLOOD_KEYS,
    FOUNDATION_KEYS,
    FloodImageHead,
    TimmUNet,
    TwoStageModel,
    apply_heads,
    build_heads,
    encoder_stats,
)


class SiameseFoundationNet(nn.Module):
    """
    Foundation U-Net (pre-event image only).

    Args:
        encoder_name: timm encoder name.
        pretrained: Load pretrained encoder weights.
        decoder_channels: Five U-Net decoder widths, deep to shallow.
    """

    def __init__(
        self,
        encoder_name: str = "efficientnet_b5",
        pretrained: bool = True,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
    ):
        super().__init__()
        self.unet = TimmUNet(encoder_name, pretrained, decoder_channels)
        self.heads = build_heads(self.unet.out_channels, FOUNDATION_KEYS, kernel_size=3)

    def forward(self, pre: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            pre: Normalised pre-event image (B, 3, H, W).

        Returns:
            Dict of foundation logits, each (B, 1, H, W).
        """
        return apply_heads(self.heads, self.unet(pre))


class SiameseFloodNet(nn.Module):
    """
    Siamese flood U-Net with a concatenation head.

    The concatenated decoder outputs (2C channels) go through a plain
    ``head_kernel_size`` convolution (2C→2C, the solution's default 'conv'
    head module) and a 1x1 classifier, split into one head per output.

    Args:
        encoder_name: timm encoder name.
        pretrained: Load pretrained encoder weights.
        decoder_channels: Five U-Net decoder widths, deep to shallow.
        head_kernel_size: Kernel size of the penultimate head convolution.
    """

    def __init__(
        self,
        encoder_name: str = "efficientnet_b5",
        pretrained: bool = True,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        head_kernel_size: int = 7,
    ):
        super().__init__()
        self.unet = TimmUNet(encoder_name, pretrained, decoder_channels)
        c = 2 * self.unet.out_channels
        self.head_conv = nn.Conv2d(c, c, head_kernel_size, padding=head_kernel_size // 2)
        self.heads = build_heads(c, FLOOD_KEYS, kernel_size=1)
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
        x = self.head_conv(x)
        out = apply_heads(self.heads, x)
        out["flood_image"] = self.flood_image(x)
        return out


class SiameseUNet(TwoStageModel):
    """
    Foundation U-Net + Siamese flood U-Net (SpaceNet 8 5th-place style).

    All four foundation outputs are native to the original solution
    (its building contact channel is dropped). The image-level flood head
    is an auxiliary head added for a uniform training interface.

    Args:
        encoder_name: timm encoder for both networks.
        pretrained: Load pretrained encoder weights.
        decoder_channels: Five U-Net decoder widths, deep to shallow.
        head_kernel_size: Kernel size of the flood head's penultimate conv.
    """

    def __init__(
        self,
        encoder_name: str = "efficientnet_b5",
        pretrained: bool = True,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        head_kernel_size: int = 7,
    ):
        super().__init__()
        self.foundation = SiameseFoundationNet(encoder_name, pretrained, decoder_channels)
        self.flood = SiameseFloodNet(
            encoder_name, pretrained, decoder_channels, head_kernel_size
        )
        self.set_normalization(*encoder_stats(self.foundation.unet.encoder))

    @torch.no_grad()
    def init_flood_from_foundation(self) -> None:
        """Copy the foundation U-Net (encoder + decoder) weights into the flood U-Net."""
        self.flood.unet.load_state_dict(self.foundation.unet.state_dict())

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
