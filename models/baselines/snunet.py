"""
SNUNet-CD: densely connected Siamese network for change detection.

Re-implementation of
    Fang, S., Li, K., Shao, J. and Li, Z., "SNUNet-CD: A Densely Connected
    Siamese Network for Change Detection of VHR Images", IEEE Geoscience
    and Remote Sensing Letters, 19, 2021.

A shared five-stage encoder processes the pre- and post-event images;
a NestedUNet (UNet++) decoder with dense skip connections from both
branches produces four full-resolution outputs, which the Ensemble
Channel Attention Module (ECAM) fuses into the change representation.
Default width is SNUNet-CD/32 (c32), trained from scratch as in the paper.

For the SpaceNet 8 contract, the ECAM change features feed the paper's
final 1x1 classifier with one channel per flood output (flooded building,
flooded road) and an image-level flood head. A light foundation path decodes the shared encoder's
pre-image features with a small U-Net decoder and 1x1 heads for
building, building border, road and junction.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn

from models.baselines.common import (
    FLOOD_KEYS,
    FOUNDATION_KEYS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    BaselineModel,
    FloodImageHead,
    UNetDecoder,
    apply_heads,
    build_heads,
    kaiming_init,
)


class NestedConvBlock(nn.Module):
    """
    SNUNet convolution unit with an inner residual connection.

    conv3x3 → BN → ReLU → conv3x3 → BN, added to the first conv output,
    followed by ReLU.

    Args:
        in_channels: Input channels.
        mid_channels: Channels after the first convolution.
        out_channels: Output channels (must equal mid_channels).
    """

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.conv1(x)
        x = self.act(self.bn1(identity))
        x = self.bn2(self.conv2(x))
        return self.act(x + identity)


class ChannelAttention(nn.Module):
    """
    Channel attention (CBAM-style) with shared MLP over avg and max pooling.

    Args:
        channels: Input channels.
        ratio: Channel reduction ratio of the bottleneck MLP.
    """

    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        hidden = max(1, channels // ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return attention weights (B, C, 1, 1) in (0, 1)."""
        avg = self.mlp(x.mean(dim=(2, 3), keepdim=True))
        mx = self.mlp(x.amax(dim=(2, 3), keepdim=True))
        return torch.sigmoid(avg + mx)


class ECAM(nn.Module):
    """
    Ensemble Channel Attention Module.

    ``M_intra = CA(sum of outputs)``, ``M_inter = CA(concat of outputs)``,
    ``ECAM = M_inter * (concat + repeat(M_intra))``.

    Args:
        channels: Channels of each deep-supervision output.
        num_outputs: Number of outputs being ensembled.
        ratio: Reduction ratio of the inter-group attention; the
            intra-group attention uses ``ratio // 4``.
    """

    def __init__(self, channels: int, num_outputs: int = 4, ratio: int = 16):
        super().__init__()
        self.num_outputs = num_outputs
        self.ca_inter = ChannelAttention(channels * num_outputs, ratio)
        self.ca_intra = ChannelAttention(channels, max(1, ratio // 4))

    def forward(self, outputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            outputs: ``num_outputs`` tensors of shape (B, C, H, W).

        Returns:
            Fused features (B, num_outputs * C, H, W).
        """
        stacked = torch.cat(list(outputs), dim=1)
        intra = self.ca_intra(sum(outputs)).repeat(1, self.num_outputs, 1, 1)
        return self.ca_inter(stacked) * (stacked + intra)


class SNUNetCD(BaselineModel):
    """
    SNUNet-CD with ECAM, adapted to the SpaceNet 8 output contract.

    The change detection network (encoder, nested decoder, ECAM) follows
    the paper exactly, including the reference design in which the
    upsampled deeper features in the nested decoder come from the
    post-event branch only. The paper's final 1x1 classifier
    (``conv_final``) predicts one channel per flood output. The
    foundation path (shared encoder on the pre image, a small U-Net
    decoder and 1x1 heads) and the image-level flood head are auxiliary
    heads added for a uniform training interface.

    Args:
        in_channels: Input channels per image.
        base_channels: Width of the first stage (32 for SNUNet-CD/32).
        foundation_channels: Decoder widths of the light foundation path,
            deep to shallow (one block per encoder skip).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        foundation_channels: Sequence[int] = (128, 64, 32, 32),
    ):
        super().__init__(IMAGENET_MEAN, IMAGENET_STD)
        f = [base_channels * 2 ** i for i in range(5)]

        self.pool = nn.MaxPool2d(2)
        self.conv0_0 = NestedConvBlock(in_channels, f[0], f[0])
        self.conv1_0 = NestedConvBlock(f[0], f[1], f[1])
        self.conv2_0 = NestedConvBlock(f[1], f[2], f[2])
        self.conv3_0 = NestedConvBlock(f[2], f[3], f[3])
        self.conv4_0 = NestedConvBlock(f[3], f[4], f[4])

        self.up1_0 = nn.ConvTranspose2d(f[1], f[1], 2, stride=2)
        self.up2_0 = nn.ConvTranspose2d(f[2], f[2], 2, stride=2)
        self.up3_0 = nn.ConvTranspose2d(f[3], f[3], 2, stride=2)
        self.up4_0 = nn.ConvTranspose2d(f[4], f[4], 2, stride=2)
        self.up1_1 = nn.ConvTranspose2d(f[1], f[1], 2, stride=2)
        self.up2_1 = nn.ConvTranspose2d(f[2], f[2], 2, stride=2)
        self.up3_1 = nn.ConvTranspose2d(f[3], f[3], 2, stride=2)
        self.up1_2 = nn.ConvTranspose2d(f[1], f[1], 2, stride=2)
        self.up2_2 = nn.ConvTranspose2d(f[2], f[2], 2, stride=2)
        self.up1_3 = nn.ConvTranspose2d(f[1], f[1], 2, stride=2)

        self.conv0_1 = NestedConvBlock(f[0] * 2 + f[1], f[0], f[0])
        self.conv1_1 = NestedConvBlock(f[1] * 2 + f[2], f[1], f[1])
        self.conv2_1 = NestedConvBlock(f[2] * 2 + f[3], f[2], f[2])
        self.conv3_1 = NestedConvBlock(f[3] * 2 + f[4], f[3], f[3])
        self.conv0_2 = NestedConvBlock(f[0] * 3 + f[1], f[0], f[0])
        self.conv1_2 = NestedConvBlock(f[1] * 3 + f[2], f[1], f[1])
        self.conv2_2 = NestedConvBlock(f[2] * 3 + f[3], f[2], f[2])
        self.conv0_3 = NestedConvBlock(f[0] * 4 + f[1], f[0], f[0])
        self.conv1_3 = NestedConvBlock(f[1] * 4 + f[2], f[1], f[1])
        self.conv0_4 = NestedConvBlock(f[0] * 5 + f[1], f[0], f[0])

        self.ecam = ECAM(f[0], num_outputs=4, ratio=16)
        self.conv_final = nn.Conv2d(4 * f[0], len(FLOOD_KEYS), 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.flood_image = FloodImageHead(4 * f[0])
        self.foundation_decoder = UNetDecoder(f, foundation_channels)
        kaiming_init(self.foundation_decoder)
        self.foundation_heads = build_heads(
            self.foundation_decoder.out_channels, FOUNDATION_KEYS, kernel_size=1
        )

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Shared Siamese encoder.

        Args:
            x: Normalised image (B, C, H, W).

        Returns:
            Features [x0_0, x1_0, x2_0, x3_0, x4_0] at strides 1..16.
        """
        x0 = self.conv0_0(x)
        x1 = self.conv1_0(self.pool(x0))
        x2 = self.conv2_0(self.pool(x1))
        x3 = self.conv3_0(self.pool(x2))
        x4 = self.conv4_0(self.pool(x3))
        return [x0, x1, x2, x3, x4]

    def change_features(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """
        Nested decoder + ECAM.

        Args:
            a: Pre-event encoder features.
            b: Post-event encoder features.

        Returns:
            ECAM change features (B, 4 * base_channels, H, W).
        """
        x0_1 = self.conv0_1(torch.cat([a[0], b[0], self.up1_0(b[1])], 1))
        x1_1 = self.conv1_1(torch.cat([a[1], b[1], self.up2_0(b[2])], 1))
        x0_2 = self.conv0_2(torch.cat([a[0], b[0], x0_1, self.up1_1(x1_1)], 1))

        x2_1 = self.conv2_1(torch.cat([a[2], b[2], self.up3_0(b[3])], 1))
        x1_2 = self.conv1_2(torch.cat([a[1], b[1], x1_1, self.up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([a[0], b[0], x0_1, x0_2, self.up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([a[3], b[3], self.up4_0(b[4])], 1))
        x2_2 = self.conv2_2(torch.cat([a[2], b[2], x2_1, self.up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([a[1], b[1], x1_1, x1_2, self.up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(
            torch.cat([a[0], b[0], x0_1, x0_2, x0_3, self.up1_3(x1_3)], 1)
        )
        return self.ecam([x0_1, x0_2, x0_3, x0_4])

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
        a = self.encode(self.normalize(pre))
        b = self.encode(self.normalize(post))

        out = apply_heads(self.foundation_heads, self.foundation_decoder(a))
        change = self.change_features(a, b)
        flood = self.conv_final(change).split(1, dim=1)
        out.update(zip(FLOOD_KEYS, flood))
        out["flood_image"] = self.flood_image(change)
        return out
