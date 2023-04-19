import logging

import torch
from torch import nn

log = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth=1e-4):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceLoss2D(nn.Module):
    """Originally implemented by Cong Gao."""

    def __init__(self, skip_bg=True):
        super(DiceLoss2D, self).__init__()

        self.skip_bg = skip_bg

    def forward(self, inputs, target):
        # Add this to numerator and denominator to avoid divide by zero when nothing is segmented
        # and ground truth is also empty (denominator term).
        # Also allow a Dice of 1 (-1) for this case (both terms).
        eps = 1.0e-4

        if self.skip_bg:
            # numerator of Dice, for each class except class 0 (background)
            numerators = 2 * torch.sum(target[:, 1:] * inputs[:, 1:], dim=(2, 3)) + eps

            # denominator of Dice, for each class except class 0 (background)
            denominators = (
                torch.sum(target[:, 1:] * target[:, 1:, :, :], dim=(2, 3))
                + torch.sum(inputs[:, 1:] * inputs[:, 1:], dim=(2, 3))
                + eps
            )

            # minus one to exclude the background class
            num_classes = inputs.shape[1] - 1
        else:
            # numerator of Dice, for each class
            numerators = 2 * torch.sum(target * inputs, dim=(2, 3)) + eps

            # denominator of Dice, for each class
            denominators = (
                torch.sum(target * target, dim=(2, 3))
                + torch.sum(inputs * inputs, dim=(2, 3))
                + eps
            )

            num_classes = inputs.shape[1]

        # Dice coefficients for each image in the batch, for each class
        dices = 1 - (numerators / denominators)

        # compute average Dice score for each image in the batch
        avg_dices = torch.sum(dices, dim=1) / num_classes

        # compute average over the batch
        return torch.mean(avg_dices)


def dice_2d(inputs, target, smooth=1e-4):
    """Dice loss for binary segmentation.

    Args:
        inputs (torch.Tensor): (C, Hmask, Wmask) or (1, Hmask, Wmask) Input tensor.
        target (torch.Tensor): Target tensor.
        smooth (float): Smoothing factor.

    Returns:
        torch.Tensor: Dice loss.
    """
    # numerator of Dice, for each region
    numerators = 2 * torch.sum(target * inputs, dim=(1, 2)) + smooth

    # denominator of Dice, for each region
    denominators = (
        torch.sum(target * target, dim=(1, 2)) + torch.sum(inputs * inputs, dim=(1, 2)) + smooth
    )

    # Dice coefficients for each image in the batch, for each class
    dices = 1 - (numerators / denominators)
    return torch.mean(dices)
