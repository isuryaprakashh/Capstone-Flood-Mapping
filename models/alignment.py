"""
Learned Pre/Post Feature Alignment.

Even after georeferenced resampling, SpaceNet 8 pre/post pairs remain offset
by a median of ~12 px (p90 ~40 px). This module estimates a dense
displacement field between pre-event and post-event feature maps from a local
correlation volume, coarse-to-fine, and warps the post features onto the pre
grid before temporal fusion.

Flow convention: flow[:, 0] = dx, flow[:, 1] = dy, in input-image pixels, such
that aligned_post(p) = post(p + flow(p)). A post image whose content is moved
by (dy, dx) relative to pre therefore has flow (dx, dy).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def local_correlation(a: torch.Tensor, b: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Cosine-similarity correlation of `a` with `b` shifted within ±radius.

    Args:
        a: Features (B, C, H, W).
        b: Features (B, C, H, W).
        radius: Maximum displacement in feature cells.

    Returns:
        Correlation volume (B, (2r+1)^2, H, W); channel k corresponds to the
        displacement (dy, dx) = divmod(k, 2r+1) - r.
    """
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    _, _, h, w = a.shape
    b = F.pad(b, (radius, radius, radius, radius))
    size = 2 * radius + 1
    return torch.cat(
        [(a * b[:, :, dy:dy + h, dx:dx + w]).sum(1, keepdim=True) for dy in range(size) for dx in range(size)],
        dim=1,
    )


def warp(features: torch.Tensor, flow_cells: torch.Tensor) -> torch.Tensor:
    """
    Backward-warp features by a flow given in feature cells.

    Args:
        features: (B, C, H, W).
        flow_cells: (B, 2, H, W) displacement (dx, dy) in cells.

    Returns:
        Warped features sampled at p + flow(p), zeros outside the map.
    """
    b, _, h, w = features.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=features.device, dtype=flow_cells.dtype),
        torch.arange(w, device=features.device, dtype=flow_cells.dtype),
        indexing="ij",
    )
    x = xs.unsqueeze(0) + flow_cells[:, 0]
    y = ys.unsqueeze(0) + flow_cells[:, 1]
    grid = torch.stack([2 * x / max(w - 1, 1) - 1, 2 * y / max(h - 1, 1) - 1], dim=-1)
    return F.grid_sample(features, grid.to(features.dtype), mode="bilinear", padding_mode="zeros", align_corners=True)


class FlowHead(nn.Module):
    """Predicts a flow update (2 ch, cells) and, optionally, a confidence logit."""

    def __init__(self, in_ch: int, hidden: int = 128, with_conf: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.flow = nn.Conv2d(hidden // 2, 2, 3, padding=1)
        self.conf = nn.Conv2d(hidden // 2, 1, 3, padding=1) if with_conf else None
        nn.init.zeros_(self.flow.weight)  # start from the identity alignment
        nn.init.zeros_(self.flow.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.net(x)
        return self.flow(h), self.conf(h) if self.conf is not None else None


class CoarseToFineAligner(nn.Module):
    """
    Coarse-to-fine feature alignment over a stride-(2..32) pyramid.

    Flow is estimated at the coarsest refinement level from a wide correlation
    window, refined at finer levels from residual correlation after warping,
    and propagated (upsampled) to all remaining levels.

    Args:
        channels: Channel count of each pyramid level (strides 2, 4, 8, 16, 32).
        levels: Strides at which flow is estimated, coarse first.
        radii: Correlation radius (cells) per estimation level.
        proj_channels: Channels of the projection used for matching.
    """

    STRIDES = (2, 4, 8, 16, 32)

    def __init__(
        self,
        channels: list[int],
        levels: tuple[int, ...] = (16, 8),
        radii: tuple[int, ...] = (4, 3),
        proj_channels: int = 64,
    ):
        super().__init__()
        assert len(levels) == len(radii)
        self.levels = levels
        self.radii = dict(zip(levels, radii))
        self.proj = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        for s, r in self.radii.items():
            c = channels[self.STRIDES.index(s)]
            self.proj[str(s)] = nn.Conv2d(c, proj_channels, 1)
            # Only the finest estimation level reports confidence
            self.heads[str(s)] = FlowHead((2 * r + 1) ** 2 + 2 * proj_channels + 2, with_conf=s == levels[-1])

    def estimate(
        self, pre_feats: list[torch.Tensor], post_feats: list[torch.Tensor]
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """
        Estimate flow at every pyramid level.

        Returns:
            (flows, confidence): flows maps stride → (B, 2, H_s, W_s) flow in
            input pixels; confidence is the logit map at the finest
            estimation level.
        """
        flow_px = None
        conf = None
        flows = {}
        for s in self.levels:
            i = self.STRIDES.index(s)
            a = self.proj[str(s)](pre_feats[i])
            b = self.proj[str(s)](post_feats[i])
            if flow_px is None:
                flow_px = a.new_zeros(a.shape[0], 2, *a.shape[2:])
            else:
                flow_px = F.interpolate(flow_px, size=a.shape[2:], mode="bilinear", align_corners=False)
            b_warped = warp(b, flow_px / s)
            corr = local_correlation(a, b_warped, self.radii[s])
            delta, level_conf = self.heads[str(s)](torch.cat([corr, a, b_warped, flow_px / s], dim=1))
            conf = level_conf if level_conf is not None else conf
            flow_px = flow_px + delta * s
            flows[s] = flow_px

        for s in self.STRIDES:
            if s not in flows:
                ref = pre_feats[self.STRIDES.index(s)]
                flows[s] = F.interpolate(flow_px, size=ref.shape[2:], mode="bilinear", align_corners=False)
        return flows, conf

    def forward(
        self, pre_feats: list[torch.Tensor], post_feats: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], dict[int, torch.Tensor], torch.Tensor]:
        """
        Align post features to pre features.

        Returns:
            (aligned_post_feats, flows, confidence).
        """
        flows, conf = self.estimate(pre_feats, post_feats)
        aligned = [warp(f, flows[s] / s) for s, f in zip(self.STRIDES, post_feats)]
        return aligned, flows, conf
