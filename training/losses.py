"""
Multi-Task Loss for the Research Models.

- building / road: BCE + soft Dice over valid pixels
- building_border / junction: BCE over valid pixels (auxiliary structure cues)
- flooded_building / flooded_road: BCE + soft Dice restricted to labelled
  object pixels (building / road): SpaceNet 8 attributes flood status to
  objects, so pixels outside objects carry no flood label and are ignored
- flood_image: BCE on the crop-level "contains flooding" flag
- alignment (optional): L1 between the estimated flow and the known offset of
  a synthetically shifted image pair, in units of the maximum offset

All terms are weighted by `weights`; heads absent from `weights` (or with
weight 0) are skipped, which is how staged training selects its heads.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.dataset import LABEL_CHANNELS

DEFAULT_WEIGHTS = {
    "building": 1.0,
    "building_border": 0.5,
    "road": 1.0,
    "junction": 0.5,
    "flooded_building": 1.0,
    "flooded_road": 1.0,
    "flood_image": 0.2,
    "alignment": 0.1,
}

DICE_HEADS = ("building", "road", "flooded_building", "flooded_road")
OBJECT_MASK = {"flooded_building": "building", "flooded_road": "road"}


def masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def masked_dice(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    prob = torch.sigmoid(logits.float()) * mask
    target = target * mask
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + smooth) / (union + smooth)).mean()


class MultiTaskLoss(nn.Module):
    """
    Args:
        weights: Per-term weights (see DEFAULT_WEIGHTS); missing or zero
            terms are not computed.
        ssl_max_px: Maximum synthetic offset, used to normalize the
            alignment loss.
        object_masked_flood: Restrict flood losses to labelled object pixels
            (False: all valid pixels, an ablation).
    """

    def __init__(self, weights: Optional[dict] = None, ssl_max_px: float = 32.0, object_masked_flood: bool = True):
        super().__init__()
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.ssl_max_px = ssl_max_px
        self.object_masked_flood = object_masked_flood

    def forward(
        self,
        outputs: dict,
        batch: dict,
        ssl_flows: Optional[dict] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            outputs: Model output dict.
            batch: Batch from SN8CropDataset (labels, valid, flood_image, ssl_*).
            ssl_flows: Flows estimated on the self-supervised pairs of the
                samples where batch['ssl'] == 1 (stride → (N, 2, h, w)).

        Returns:
            Dict with 'total' and one detached entry per computed term.
        """
        labels = batch["labels"]
        valid = batch["valid"]
        terms = {}

        for i, head in enumerate(LABEL_CHANNELS):
            w = self.weights.get(head, 0.0)
            if not w or head not in outputs:
                continue
            target = labels[:, i:i + 1]
            mask = valid
            if head in OBJECT_MASK and self.object_masked_flood:
                j = LABEL_CHANNELS.index(OBJECT_MASK[head])
                mask = valid * labels[:, j:j + 1]
            loss = masked_bce(outputs[head], target, mask)
            if head in DICE_HEADS:
                loss = loss + masked_dice(outputs[head], target, mask)
            terms[head] = w * loss

        w = self.weights.get("flood_image", 0.0)
        if w and "flood_image" in outputs:
            terms["flood_image"] = w * F.binary_cross_entropy_with_logits(
                outputs["flood_image"].float(), batch["flood_image"]
            )

        w = self.weights.get("alignment", 0.0)
        if w and ssl_flows:
            target = batch["ssl_flow"][batch["ssl"] > 0]  # (N, 2) as (dx, dy)
            per_level = [
                (flow.float() - target[:, :, None, None]).abs().mean() / self.ssl_max_px
                for flow in ssl_flows.values()
            ]
            terms["alignment"] = w * torch.stack(per_level).mean()

        total = torch.stack(list(terms.values())).sum() if terms else labels.new_zeros(())
        out = {k: v.detach() for k, v in terms.items()}
        out["total"] = total
        return out
