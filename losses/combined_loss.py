"""
Combined Loss Function for Multi-Class Segmentation.

Implements a weighted combination of:
  - Binary Cross-Entropy (BCE) Loss
  - Dice Loss (spatial overlap)
  - Focal Loss (class imbalance handling)

All losses operate per-class and are averaged across classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Differentiable Dice Loss with optional class weighting.
    """

    def __init__(self, smooth: float = 1.0, per_class: bool = True, class_weights: torch.Tensor = None):
        super().__init__()
        self.smooth = smooth
        self.per_class = per_class
        self.register_buffer("class_weights", class_weights)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        B, C, H, W = probs.shape

        if self.per_class:
            probs_flat = probs.reshape(B, C, -1)
            targets_flat = targets.reshape(B, C, -1)

            intersection = (probs_flat * targets_flat).sum(dim=2)
            union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            loss = 1.0 - dice  # (B, C)

            if self.class_weights is not None:
                w = self.class_weights.to(loss.device).view(1, -1)
                return (loss * w).sum() / (w.sum() * B + 1e-8)
            return loss.mean()
        else:
            probs_flat = probs.reshape(-1)
            targets_flat = targets.reshape(-1)

            intersection = (probs_flat * targets_flat).sum()
            union = probs_flat.sum() + targets_flat.sum()

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme class imbalance in flood mapping.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, class_weights: torch.Tensor = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer("class_weights", class_weights)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)

        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * bce

        if self.class_weights is not None:
            w = self.class_weights.to(focal_loss.device).reshape(1, -1, 1, 1)
            focal_loss = focal_loss * w
            return focal_loss.sum() / (w.sum() * focal_loss.shape[0] * focal_loss.shape[2] * focal_loss.shape[3] + 1e-8)

        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Optimized Combined Loss Function: α·Dice + β·Focal + γ·BCE with Class Weighting.
    
    Default class weights [1.0, 1.2, 2.0, 2.5, 2.5] prioritize hard-to-detect
    flooded roads and flooded buildings for maximum mIoU score.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        bce_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_smooth: float = 1.0,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight

        # Default class weighting if none provided: [bldg, road, flood, f_bldg, f_road]
        if class_weights is None:
            class_weights = torch.tensor([1.0, 1.2, 2.0, 2.5, 2.5], dtype=torch.float32)

        self.register_buffer("class_weights", class_weights)

        self.dice_loss = DiceLoss(smooth=dice_smooth, per_class=True, class_weights=class_weights)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, class_weights=class_weights)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute weighted multi-task loss."""
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device).reshape(1, -1, 1, 1)
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, weight=weights
            )
        else:
            bce = F.binary_cross_entropy_with_logits(logits, targets)

        total = (
            self.dice_weight * dice
            + self.focal_weight * focal
            + self.bce_weight * bce
        )

        return {
            "total": total,
            "dice": dice.detach(),
            "focal": focal.detach(),
            "bce": bce.detach(),
        }
