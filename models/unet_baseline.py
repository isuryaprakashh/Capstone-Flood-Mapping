"""
U-Net Baseline Model for SpaceNet 8 Flood Mapping.

Standard encoder-decoder segmentation architecture with a
pretrained ResNet34 encoder and skip connections. Supports
multi-channel input (pre+post concatenation = 6 channels)
and 5-class output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block: Conv → BN → ReLU → Conv → BN → ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EncoderBlock(nn.Module):
    """Encoder block: ConvBlock → MaxPool."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    """Decoder block: Upsample → Concat skip → ConvBlock."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        x = self.up(x)

        # Handle size mismatch due to odd dimensions
        if x.shape != skip.shape:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNetEncoder(nn.Module):
    """
    ResNet34-based encoder that extracts multi-scale features.

    Uses torchvision's pretrained ResNet34 and extracts features
    at 4 resolution levels (1/2, 1/4, 1/8, 1/16).
    """

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        import torchvision.models as models

        # Load pretrained ResNet34
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet34(weights=weights)

        # Modify first conv layer for non-3-channel input
        if in_channels != 3:
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # Initialize with pretrained weights where possible
            if pretrained:
                with torch.no_grad():
                    # Copy pretrained weights for first 3 channels
                    orig_weight = resnet.conv1.weight.data
                    if in_channels > 3:
                        # Repeat weights for extra channels
                        repeat_count = (in_channels + 2) // 3
                        expanded = orig_weight.repeat(1, repeat_count, 1, 1)
                        self.conv1.weight.data = expanded[:, :in_channels, :, :]
                    else:
                        self.conv1.weight.data = orig_weight[:, :in_channels, :, :]
        else:
            self.conv1 = resnet.conv1

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64 channels, stride 4
        self.layer2 = resnet.layer2  # 128 channels, stride 8
        self.layer3 = resnet.layer3  # 256 channels, stride 16
        self.layer4 = resnet.layer4  # 512 channels, stride 32

        # Channel counts at each level
        self.channels = [64, 64, 128, 256, 512]

    def forward(
        self, x: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Extract multi-scale features.

        Returns:
            List of feature maps at 5 scales:
            [stem(1/2), layer1(1/4), layer2(1/8), layer3(1/16), layer4(1/32)]
        """
        features = []

        # Stem: 1/2 resolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)  # 64 channels

        x = self.maxpool(x)

        # Layer blocks
        x = self.layer1(x)
        features.append(x)  # 64 channels, 1/4

        x = self.layer2(x)
        features.append(x)  # 128 channels, 1/8

        x = self.layer3(x)
        features.append(x)  # 256 channels, 1/16

        x = self.layer4(x)
        features.append(x)  # 512 channels, 1/32

        return features


class UNetBaseline(nn.Module):
    """
    U-Net Baseline Model with ResNet34 Encoder.

    Architecture:
        - ResNet34 pretrained encoder (4 resolution levels)
        - Decoder with transposed convolutions and skip connections
        - Multi-class segmentation head (5 channels)

    Args:
        in_channels: Number of input channels (3 for single image,
                     6 for pre+post concatenation).
        num_classes: Number of output segmentation classes.
        pretrained: Use ImageNet pretrained encoder weights.
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 5,
        pretrained: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder
        self.encoder = ResNetEncoder(in_channels, pretrained)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder
        self.decoder4 = DecoderBlock(1024, 512, 512)
        self.decoder3 = DecoderBlock(512, 256, 256)
        self.decoder2 = DecoderBlock(256, 128, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)

        # Final upsampling + head
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            ConvBlock(32 + 64, 32),  # +64 from stem skip
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(
        self,
        pre_image: torch.Tensor,
        post_image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pre_image: Pre-event image tensor (B, 3, H, W).
            post_image: Post-event image tensor (B, 3, H, W).

        Returns:
            Segmentation logits (B, num_classes, H, W).
        """
        # Concatenate pre and post images along channel dim
        x = torch.cat([pre_image, post_image], dim=1)  # (B, 6, H, W)

        # Encode
        features = self.encoder(x)
        # features: [stem, layer1, layer2, layer3, layer4]

        # Bottleneck
        bottleneck = self.bottleneck(features[4])

        # Decode with skip connections
        d4 = self.decoder4(bottleneck, features[4])  # skip from layer4
        d3 = self.decoder3(d4, features[3])           # skip from layer3
        d2 = self.decoder2(d3, features[2])           # skip from layer2
        d1 = self.decoder1(d2, features[1])           # skip from layer1

        # Final upsample + stem skip
        up = self.final_up(d1)
        if up.shape[2:] != features[0].shape[2:]:
            up = F.interpolate(
                up, size=features[0].shape[2:],
                mode="bilinear", align_corners=False,
            )
        up = torch.cat([up, features[0]], dim=1)
        logits = self.final_conv(up)

        # Upsample to input resolution if needed
        input_h, input_w = pre_image.shape[2:]
        if logits.shape[2:] != (input_h, input_w):
            logits = F.interpolate(
                logits,
                size=(input_h, input_w),
                mode="bilinear",
                align_corners=False,
            )

        return logits

    def predict(
        self,
        pre_image: torch.Tensor,
        post_image: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Run inference and return binary predictions.

        Args:
            pre_image: Pre-event image (B, 3, H, W).
            post_image: Post-event image (B, 3, H, W).
            threshold: Binarization threshold.

        Returns:
            Binary predictions (B, num_classes, H, W).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(pre_image, post_image)
            probs = torch.sigmoid(logits)
            return (probs > threshold).float()
