"""
Feature Backbones for the SpaceNet 8 Research Models.

Every backbone takes raw RGB in [0, 1], normalizes it internally with its own
pretraining statistics, and returns a 5-level feature pyramid at strides
(2, 4, 8, 16, 32) whose channel counts are listed in `self.channels`.

- CNNBackbone: any timm CNN encoder (ResNet, EfficientNet, ConvNeXt), fine-tuned.
- DinoV3SatBackbone: frozen DINOv3 ViT-L/16 pretrained on Maxar satellite
  imagery (SAT-493M) for semantics, plus a trainable ResNet-18 detail stream
  for the high-resolution levels the ViT (patch 16) cannot provide.
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
STRIDES = (2, 4, 8, 16, 32)


class Normalize(nn.Module):
    """Channel-wise (x - mean) / std with buffers that follow .to(device)."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def conv_bn_relu(in_ch: int, out_ch: int, kernel: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CNNBackbone(nn.Module):
    """
    timm CNN encoder exposed as a stride-(2..32) pyramid.

    Encoders without a stride-2 stage (e.g. ConvNeXt) get a small convolutional
    stem on the image for that level.

    Args:
        name: timm model name (e.g. 'resnet34', 'efficientnet_b5',
            'convnext_tiny.dinov3_lvd1689m').
        pretrained: Load pretrained weights.
    """

    def __init__(self, name: str = "resnet34", pretrained: bool = True):
        super().__init__()
        self.encoder = timm.create_model(name, features_only=True, pretrained=pretrained)
        cfg = self.encoder.pretrained_cfg
        self.normalize = Normalize(cfg.get("mean", IMAGENET_MEAN), cfg.get("std", IMAGENET_STD))

        info = self.encoder.feature_info
        by_stride = {r: (i, c) for i, (r, c) in enumerate(zip(info.reduction(), info.channels()))}
        self.indices = {s: by_stride[s][0] for s in STRIDES if s in by_stride}
        self.stem = None
        if 2 not in self.indices:
            self.stem = nn.Sequential(conv_bn_relu(3, 32), conv_bn_relu(32, 64))
        self.channels = [64 if s == 2 and self.stem is not None else by_stride[s][1] for s in STRIDES]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        xn = self.normalize(x)
        feats = self.encoder(xn)
        out = []
        for s in STRIDES:
            if s == 2 and self.stem is not None:
                out.append(self.stem(F.avg_pool2d(xn, 2)))
            else:
                out.append(feats[self.indices[s]])
        return out


class DinoV3SatBackbone(nn.Module):
    """
    Frozen DINOv3 ViT-L/16 (SAT-493M) + trainable ResNet-18 detail stream.

    Intermediate ViT blocks are concatenated into a stride-16 semantic map,
    from which a simple feature pyramid (ViTDet-style) derives strides 8 and
    32 and, together with the detail stream, strides 2 and 4.

    Args:
        pretrained: Load DINOv3 and ResNet-18 pretrained weights.
        freeze: Keep the ViT frozen (no gradients, eval-mode norms).
        layers: ViT blocks whose outputs form the semantic map.
        width: Channels of the stride-8/16/32 pyramid levels.
    """

    VIT_NAME = "vit_large_patch16_dinov3.sat493m"

    def __init__(
        self,
        pretrained: bool = True,
        freeze: bool = True,
        layers: tuple[int, ...] = (5, 11, 17, 23),
        width: int = 256,
    ):
        super().__init__()
        self.freeze = freeze
        self.layers = list(layers)
        self.vit = timm.create_model(self.VIT_NAME, pretrained=pretrained, num_classes=0, dynamic_img_size=True)
        self.vit_norm = Normalize(self.vit.pretrained_cfg["mean"], self.vit.pretrained_cfg["std"])
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

        self.detail = timm.create_model("resnet18", features_only=True, pretrained=pretrained, out_indices=(0, 1, 2))
        self.detail_norm = Normalize(IMAGENET_MEAN, IMAGENET_STD)
        d2, d4, d8 = self.detail.feature_info.channels()  # strides 2, 4, 8

        vit_dim = self.vit.embed_dim
        self.semantic = conv_bn_relu(vit_dim * len(self.layers), 2 * width, kernel=1)
        self.to16 = conv_bn_relu(2 * width, width, kernel=1)
        self.to32 = nn.Sequential(nn.MaxPool2d(2), conv_bn_relu(2 * width, width))
        self.up8 = nn.ConvTranspose2d(2 * width, width, 2, stride=2)
        self.fuse8 = conv_bn_relu(width + d8, width)
        self.up4 = nn.ConvTranspose2d(width, width // 2, 2, stride=2)
        self.fuse4 = conv_bn_relu(width // 2 + d4, width // 2)
        self.fuse2 = conv_bn_relu(d2, 64)

        self.channels = [64, width // 2, width, width, width]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.vit.eval()  # frozen LayerNorm/dropout behave as at inference
        return self

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze):
            blocks = self.vit.forward_intermediates(
                self.vit_norm(x), indices=self.layers, output_fmt="NCHW", intermediates_only=True
            )
        sem = self.semantic(torch.cat(blocks, dim=1))  # stride 16
        s2, s4, s8 = self.detail(self.detail_norm(x))

        p16 = self.to16(sem)
        p32 = self.to32(sem)
        p8 = self.fuse8(torch.cat([self.up8(sem), s8], dim=1))
        p4 = self.fuse4(torch.cat([self.up4(p8), s4], dim=1))
        p2 = self.fuse2(s2)
        return [p2, p4, p8, p16, p32]


def build_backbone(name: str, pretrained: bool = True, **kwargs) -> nn.Module:
    """
    Build a backbone by name: 'dinov3_sat' or any timm CNN encoder name.
    """
    if name == "dinov3_sat":
        return DinoV3SatBackbone(pretrained=pretrained, **kwargs)
    return CNNBackbone(name, pretrained=pretrained)
