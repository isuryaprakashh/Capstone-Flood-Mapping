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
    Differentiable Dice Loss for binary/multi-class segmentation.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice

    Args:
        smooth: Smoothing factor to avoid division by zero.
        per_class: If True, compute Dice per class and average.
    """

    def __init__(self, smooth: float = 1.0, per_class: bool = True):
        super().__init__()
        self.smooth = smooth
        self.per_class = per_class

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: Model output (B, C, H, W) — raw logits.
            targets: Ground truth (B, C, H, W) — binary masks.

        Returns:
            Scalar Dice loss.
        """
        probs = torch.sigmoid(logits)
        B, C, H, W = probs.shape

        if self.per_class:
            # Flatten spatial dims: (B, C, H*W)
            probs_flat = probs.reshape(B, C, -1)
            targets_flat = targets.reshape(B, C, -1)

            intersection = (probs_flat * targets_flat).sum(dim=2)
            union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            loss = 1.0 - dice  # (B, C)
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
    Focal Loss for addressing class imbalance.

    FL(p) = -α * (1 - p)^γ * log(p)

    Down-weights easy examples and focuses on hard negatives,
    which is critical for satellite imagery where flood pixels
    are a small fraction of the total.

    Args:
        alpha: Balancing factor (default: 0.25).
        gamma: Focusing parameter (default: 2.0).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: Model output (B, C, H, W) — raw logits.
            targets: Ground truth (B, C, H, W) — binary masks.

        Returns:
            Scalar Focal loss.
        """
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)

        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * bce

        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function: α·Dice + β·Focal + γ·BCE.

    This combination provides:
    - Dice: Global overlap optimization (fights class imbalance)
    - Focal: Hard example mining (focuses on difficult pixels)
    - BCE: Stable per-pixel classification baseline

    Args:
        dice_weight: Weight for Dice loss (default: 1.0).
        focal_weight: Weight for Focal loss (default: 1.0).
        bce_weight: Weight for BCE loss (default: 0.5).
        focal_alpha: Focal loss alpha parameter.
        focal_gamma: Focal loss gamma parameter.
        dice_smooth: Dice loss smoothing factor.
        class_weights: Optional per-class weights (tensor of shape [C]).
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

        self.dice_loss = DiceLoss(smooth=dice_smooth, per_class=True)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else None,
        )

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            logits: Model predictions (B, C, H, W) — raw logits.
            targets: Ground truth masks (B, C, H, W) — binary.

        Returns:
            Dictionary with 'total', 'dice', 'focal', 'bce' loss values.
        """
        # Compute individual losses
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)

        # BCE with optional class weights
        if self.class_weights is not None:
            # Reshape class_weights to (1, C, 1, 1) for broadcasting
            weights = self.class_weights.reshape(1, -1, 1, 1)
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, weight=weights
            )
        else:
            bce = F.binary_cross_entropy_with_logits(logits, targets)

        # Weighted combination
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
