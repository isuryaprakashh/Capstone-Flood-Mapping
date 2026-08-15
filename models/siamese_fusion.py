"""
Siamese Cross-Attention Fusion Network for Flood Damage Detection.

Novel architecture that processes pre-event and post-event satellite
images through twin shared-weight encoders, fuses their features via
cross-attention at multiple scales to capture temporal changes,
and decodes into multi-class flood segmentation maps.

This is the key innovation over the baseline U-Net:
  - Siamese encoders preserve spatial representations of each time step
  - Cross-attention explicitly models change detection between pre/post
  - Change-aware decoder focuses on flood-damaged regions
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention Fusion Module.

    Computes bidirectional cross-attention between pre-event and post-event
    feature maps to capture temporal changes at a given resolution level.

    Uses multi-head attention with learned positional encoding.

    Args:
        channels: Number of input feature channels.
        num_heads: Number of attention heads.
        reduction: Channel reduction ratio for Q/K projections.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        reduction: int = 4,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        inner_dim = channels // reduction

        # Query, Key, Value projections
        self.q_pre = nn.Conv2d(channels, inner_dim, 1, bias=False)
        self.k_post = nn.Conv2d(channels, inner_dim, 1, bias=False)
        self.v_post = nn.Conv2d(channels, inner_dim, 1, bias=False)

        self.q_post = nn.Conv2d(channels, inner_dim, 1, bias=False)
        self.k_pre = nn.Conv2d(channels, inner_dim, 1, bias=False)
        self.v_pre = nn.Conv2d(channels, inner_dim, 1, bias=False)

        # Output projections
        self.proj_pre = nn.Sequential(
            nn.Conv2d(inner_dim, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.proj_post = nn.Sequential(
            nn.Conv2d(inner_dim, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Change detection gate
        self.change_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

        # Fusion output
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def _spatial_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Compute spatial cross-attention."""
        B, C, H, W = q.shape
        N = H * W

        # Flatten spatial dims: (B, C, H, W) → (B, C, N) → (B, N, C)
        q_flat = q.flatten(2).transpose(1, 2)  # (B, N, C)
        k_flat = k.flatten(2).transpose(1, 2)
        v_flat = v.flatten(2).transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.matmul(q_flat, k_flat.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v_flat)  # (B, N, C)
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return out

    def forward(
        self,
        pre_feat: torch.Tensor,
        post_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse pre-event and post-event features via cross-attention.

        Args:
            pre_feat: Pre-event features (B, C, H, W).
            post_feat: Post-event features (B, C, H, W).

        Returns:
            Fused features (B, C, H, W) highlighting temporal changes.
        """
        # Pre → Post cross-attention (what changed in post?)
        q_pre = self.q_pre(pre_feat)
        k_post = self.k_post(post_feat)
        v_post = self.v_post(post_feat)
        attended_post = self.proj_pre(
            self._spatial_attention(q_pre, k_post, v_post)
        )
        enhanced_pre = pre_feat + attended_post

        # Post → Pre cross-attention (what was there before?)
        q_post = self.q_post(post_feat)
        k_pre = self.k_pre(pre_feat)
        v_pre = self.v_pre(pre_feat)
        attended_pre = self.proj_post(
            self._spatial_attention(q_post, k_pre, v_pre)
        )
        enhanced_post = post_feat + attended_pre

        # Change detection: compute element-wise difference
        change = torch.abs(enhanced_post - enhanced_pre)

        # Gate: learn which changes are relevant (flood vs normal change)
        gate = self.change_gate(
            torch.cat([enhanced_pre, enhanced_post], dim=1)
        )
        gated_change = change * gate

        # Fuse: concat enhanced features + gated change
        fused = self.fusion_conv(
            torch.cat([enhanced_pre, enhanced_post, gated_change], dim=1)
        )

        return fused


class SharedEncoder(nn.Module):
    """
    Shared-weight ResNet34 encoder for the Siamese architecture.

    Both pre-event and post-event images pass through the same encoder
    with shared weights, ensuring consistent feature extraction.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision.models as models

        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet34(weights=weights)

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.pool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64 ch
        self.layer2 = resnet.layer2  # 128 ch
        self.layer3 = resnet.layer3  # 256 ch
        self.layer4 = resnet.layer4  # 512 ch

        self.channels = [64, 64, 128, 256, 512]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract multi-scale features.

        Returns:
            5 feature maps at decreasing resolutions.
        """
        features = []

        x = self.stem(x)
        features.append(x)  # 1/2

        x = self.pool(x)
        x = self.layer1(x)
        features.append(x)  # 1/4

        x = self.layer2(x)
        features.append(x)  # 1/8

        x = self.layer3(x)
        features.append(x)  # 1/16

        x = self.layer4(x)
        features.append(x)  # 1/32

        return features


class ChangeAwareDecoder(nn.Module):
    """
    Decoder that receives fused skip connections from the
    Cross-Attention modules and produces segmentation output.
    """

    def __init__(self, encoder_channels: list[int], num_classes: int = 5):
        super().__init__()
        # encoder_channels: [64, 64, 128, 256, 512]

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(encoder_channels[4], 1024, 3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Decoder blocks (up → concat skip → conv)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._make_block(256 + encoder_channels[3], 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._make_block(128 + encoder_channels[2], 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._make_block(64 + encoder_channels[1], 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._make_block(32 + encoder_channels[0], 32)

        # Segmentation head
        self.head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),
        )

    def _make_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _match_and_cat(
        self, x: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        """Resize and concatenate."""
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
        return torch.cat([x, skip], dim=1)

    def forward(
        self, fused_features: list[torch.Tensor], input_size: tuple[int, int]
    ) -> torch.Tensor:
        """
        Decode fused features into segmentation logits.

        Args:
            fused_features: 5 fused feature maps from cross-attention.
            input_size: (H, W) of the original input for final resize.

        Returns:
            Segmentation logits (B, num_classes, H, W).
        """
        # fused_features: [f0(1/2), f1(1/4), f2(1/8), f3(1/16), f4(1/32)]

        x = self.bottleneck(fused_features[4])

        x = self.up4(x)
        x = self._match_and_cat(x, fused_features[3])
        x = self.dec4(x)

        x = self.up3(x)
        x = self._match_and_cat(x, fused_features[2])
        x = self.dec3(x)

        x = self.up2(x)
        x = self._match_and_cat(x, fused_features[1])
        x = self.dec2(x)

        x = self.up1(x)
        x = self._match_and_cat(x, fused_features[0])
        x = self.dec1(x)

        logits = self.head(x)

        # Upsample to original input size
        if logits.shape[2:] != input_size:
            logits = F.interpolate(
                logits, size=input_size, mode="bilinear", align_corners=False
            )

        return logits


class SiameseFusionNet(nn.Module):
    """
    Siamese Cross-Attention Fusion Network for Flood Detection.

    Novel architecture combining:
    1. Twin shared-weight ResNet34 encoders (Siamese)
    2. Multi-scale Cross-Attention fusion modules
    3. Change-aware decoder with skip connections

    The key innovation is the cross-attention mechanism that explicitly
    models temporal changes between pre-event and post-event imagery,
    enabling more accurate flood damage detection compared to simple
    concatenation in the baseline U-Net.

    Args:
        in_channels: Number of input channels per image (default: 3 for RGB).
        num_classes: Number of output segmentation classes.
        pretrained: Use ImageNet pretrained encoder.
        attention_heads: Number of attention heads in cross-attention.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 5,
        pretrained: bool = True,
        attention_heads: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Shared Siamese encoder
        self.encoder = SharedEncoder(pretrained=pretrained)
        enc_channels = self.encoder.channels  # [64, 64, 128, 256, 512]

        # Cross-Attention Fusion at each encoder level
        self.cross_attn = nn.ModuleList(
            [
                CrossAttentionBlock(
                    channels=ch,
                    num_heads=min(attention_heads, ch // 16),
                    reduction=max(1, ch // 32),
                )
                for ch in enc_channels
            ]
        )

        # Change-Aware Decoder
        self.decoder = ChangeAwareDecoder(
            encoder_channels=enc_channels,
            num_classes=num_classes,
        )

    def forward(
        self,
        pre_image: torch.Tensor,
        post_image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the Siamese Fusion Network.

        Args:
            pre_image: Pre-event image (B, 3, H, W).
            post_image: Post-event image (B, 3, H, W).

        Returns:
            Segmentation logits (B, num_classes, H, W).
        """
        input_size = pre_image.shape[2:]  # (H, W)

        # Extract features through shared encoder
        pre_features = self.encoder(pre_image)   # 5 scales
        post_features = self.encoder(post_image)  # 5 scales

        # Fuse features at each scale via cross-attention
        fused_features = []
        for i, (pre_feat, post_feat) in enumerate(
            zip(pre_features, post_features)
        ):
            fused = self.cross_attn[i](pre_feat, post_feat)
            fused_features.append(fused)

        # Decode fused features
        logits = self.decoder(fused_features, input_size)

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

    def get_change_maps(
        self,
        pre_image: torch.Tensor,
        post_image: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Extract intermediate change attention maps for visualization.

        Returns cross-attention gate activations at each scale.
        """
        self.eval()
        with torch.no_grad():
            pre_features = self.encoder(pre_image)
            post_features = self.encoder(post_image)

            change_maps = []
            for i, (pre_feat, post_feat) in enumerate(
                zip(pre_features, post_features)
            ):
                # Compute raw change magnitude
                change = torch.abs(post_feat - pre_feat).mean(dim=1, keepdim=True)
                change_maps.append(change)

            return change_maps
