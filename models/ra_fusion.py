"""
Registration-Aware Siamese Fusion Network (RA-Fusion) for SpaceNet 8.

A shared backbone encodes the pre- and post-event images. Building and road
structure (footprints, instance borders, centerlines, junctions) is decoded
from the pre-event features alone. For flood attribution, the post-event
features are first aligned onto the pre-event grid by a learned
coarse-to-fine displacement field (models/alignment.py), then fused with the
pre-event features by the change-gated cross-attention block of the earlier
Siamese model, and decoded into flooded-building / flooded-road maps plus an
image-level "any flood" logit.

Ablation switches: backbone ('dinov3_sat' or a timm CNN), align (learned
alignment on/off) and fusion ('gated' cross-attention or plain 'concat').
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.alignment import CoarseToFineAligner
from models.backbones import build_backbone, conv_bn_relu
from models.siamese_fusion import CrossAttentionBlock

FOUNDATION_HEADS = ("building", "building_border", "road", "junction")
FLOOD_HEADS = ("flooded_building", "flooded_road")


class UNetDecoder(nn.Module):
    """Top-down decoder over a stride-(2..32) pyramid; returns stride-2 features."""

    def __init__(self, channels: list[int], widths: tuple[int, ...] = (256, 128, 64, 32)):
        super().__init__()
        c2, c4, c8, c16, c32 = channels
        skips = (c16, c8, c4, c2)
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        in_ch = c32
        for skip, width in zip(skips, widths):
            self.ups.append(nn.ConvTranspose2d(in_ch, width, 2, stride=2))
            self.blocks.append(nn.Sequential(conv_bn_relu(width + skip, width), conv_bn_relu(width, width)))
            in_ch = width
        self.out_channels = widths[-1]

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        x = feats[4]
        for up, block, skip in zip(self.ups, self.blocks, (feats[3], feats[2], feats[1], feats[0])):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return x


class ConcatFusion(nn.Module):
    """Plain fusion baseline: conv over [pre, post, |pre - post|]."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(conv_bn_relu(3 * channels, channels), conv_bn_relu(channels, channels))

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([pre, post, torch.abs(pre - post)], dim=1))


class RAFusionNet(nn.Module):
    """
    Registration-aware Siamese fusion network.

    Args:
        backbone: 'dinov3_sat' (frozen satellite ViT-L + detail CNN) or a timm
            CNN encoder name (e.g. 'convnext_tiny.dinov3_lvd1689m').
        pretrained: Load pretrained backbone weights.
        align: Use learned feature alignment before fusion.
        fusion: 'gated' (change-gated cross-attention) or 'concat'.
        align_levels: Strides at which flow is estimated (coarse first).
        align_radii: Correlation radius per estimation level, in cells.
    """

    def __init__(
        self,
        backbone: str = "dinov3_sat",
        pretrained: bool = True,
        align: bool = True,
        fusion: str = "gated",
        align_levels: tuple[int, ...] = (16, 8),
        align_radii: tuple[int, ...] = (4, 3),
        **backbone_kwargs,
    ):
        super().__init__()
        self.backbone = build_backbone(backbone, pretrained=pretrained, **backbone_kwargs)
        channels = self.backbone.channels
        # Weights restored from pretraining rather than checkpoints (frozen ViT)
        self.frozen_prefixes = ("backbone.vit.",) if getattr(self.backbone, "freeze", False) else ()

        self.aligner = CoarseToFineAligner(channels, align_levels, align_radii) if align else None
        if fusion == "gated":
            self.fusion = nn.ModuleList([
                CrossAttentionBlock(c, num_heads=min(4, max(1, c // 16))) for c in channels
            ])
        elif fusion == "concat":
            self.fusion = nn.ModuleList([ConcatFusion(c) for c in channels])
        else:
            raise ValueError(f"Unknown fusion '{fusion}'")

        self.foundation_decoder = UNetDecoder(channels)
        self.flood_decoder = UNetDecoder(channels)
        w = self.foundation_decoder.out_channels
        self.foundation_heads = nn.ModuleDict({k: nn.Conv2d(w, 1, 1) for k in FOUNDATION_HEADS})
        self.flood_heads = nn.ModuleDict({k: nn.Conv2d(w, 1, 1) for k in FLOOD_HEADS})
        self.flood_image_head = nn.Sequential(
            nn.Linear(2 * channels[4], 128), nn.ReLU(inplace=True), nn.Linear(128, 1)
        )

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            pre: Pre-event RGB in [0, 1], (B, 3, H, W), H and W divisible by 32.
            post: Post-event RGB in [0, 1], same shape.

        Returns:
            Dict of logits at input resolution: building, building_border,
            road, junction, flooded_building, flooded_road (B, 1, H, W);
            flood_image (B, 1); and, when aligning, 'flows' (stride → flow in
            input pixels) and 'align_conf' (B, 1, H, W) logits.
        """
        size = pre.shape[2:]
        pre_feats = self.backbone(pre)
        post_feats = self.backbone(post)

        out = {}
        if self.aligner is not None:
            post_feats, flows, conf = self.aligner(pre_feats, post_feats)
            out["flows"] = flows
            out["align_conf"] = F.interpolate(conf, size=size, mode="bilinear", align_corners=False)

        fused = [f(a, b) for f, a, b in zip(self.fusion, pre_feats, post_feats)]

        found = self.foundation_decoder(pre_feats)
        flood = self.flood_decoder(fused)
        for k, head in self.foundation_heads.items():
            out[k] = F.interpolate(head(found), size=size, mode="bilinear", align_corners=False)
        for k, head in self.flood_heads.items():
            out[k] = F.interpolate(head(flood), size=size, mode="bilinear", align_corners=False)

        deep = fused[4]
        pooled = torch.cat([F.adaptive_avg_pool2d(deep, 1), F.adaptive_max_pool2d(deep, 1)], dim=1).flatten(1)
        out["flood_image"] = self.flood_image_head(pooled)
        return out

    def estimate_flow(self, img_a: torch.Tensor, img_b: torch.Tensor) -> dict[int, torch.Tensor]:
        """
        Flow aligning img_b onto img_a (used by the self-supervised alignment
        loss on synthetically shifted image pairs).
        """
        if self.aligner is None:
            raise RuntimeError("Model was built with align=False")
        flows, _ = self.aligner.estimate(self.backbone(img_a), self.backbone(img_b))
        return flows
