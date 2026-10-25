"""Loss functions for SpaceNet 8 Flood Mapping."""

from legacy.losses.combined_loss import CombinedLoss, DiceLoss, FocalLoss

__all__ = ["CombinedLoss", "DiceLoss", "FocalLoss"]
