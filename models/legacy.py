"""
Adapters for the earlier models (v0) on the research training interface.

The early-fusion U-Net (models/unet_baseline.py) and the Siamese
cross-attention network (models/siamese_fusion.py) are kept architecturally
unchanged; only their output layer is widened to the research targets, so
they can be retrained on the corrected pipeline and compared fairly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import IMAGENET_MEAN, IMAGENET_STD, Normalize
from models.siamese_fusion import SiameseFusionNet
from models.unet_baseline import UNetBaseline

OUTPUT_CHANNELS = (
    "building", "building_border", "road", "junction",
    "flooded_building", "flooded_road", "flood_map",
)


class LegacyAdapter(nn.Module):
    """
    Wraps a v0 model so it returns the research output dict.

    The image-level flood logit is the spatial max of an extra 'flood_map'
    channel.

    Args:
        kind: 'unet_v0' or 'siamese_v0'.
        pretrained: ImageNet-pretrained ResNet34 encoder.
    """

    def __init__(self, kind: str = "siamese_v0", pretrained: bool = True):
        super().__init__()
        n = len(OUTPUT_CHANNELS)
        if kind == "unet_v0":
            self.net = UNetBaseline(in_channels=6, num_classes=n, pretrained=pretrained)
        elif kind == "siamese_v0":
            self.net = SiameseFusionNet(in_channels=3, num_classes=n, pretrained=pretrained)
        else:
            raise ValueError(f"Unknown legacy model '{kind}'")
        self.normalize = Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.net(self.normalize(pre), self.normalize(post))
        out = {k: logits[:, i:i + 1] for i, k in enumerate(OUTPUT_CHANNELS[:-1])}
        out["flood_image"] = F.adaptive_max_pool2d(logits[:, -1:], 1).flatten(1)
        return out
