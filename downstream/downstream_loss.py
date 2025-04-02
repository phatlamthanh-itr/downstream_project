import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth                    # To avoid division by zero

    def forward(self, preds, targets):
        """
            preds:, shape (batch, time_step)
            targets: Ground truth labels (0 or 1), shape (batch, time_step)
        """
        preds = torch.sigmoid(preds)                # Convert logits to probabilities
        
        intersection = (preds * targets).sum(dim=1)  
        union = preds.sum(dim=1) + targets.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score.mean()  # Average over batch

        return dice_loss
    

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        """
        Focal Loss for binary segmentation tasks.

        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
            inputs (torch.Tensor): (batch, length).
            targets (torch.Tensor): (batch, length).
        """
        # Compute probabilities using sigmoid
        probs = torch.sigmoid(inputs)
        log_probs = torch.log(probs + 1e-10)  
        log_probs_neg = torch.log(1 - probs + 1e-10)

        # Compute focal loss 
        loss = -self.alpha * (targets * ((1 - probs) ** self.gamma) * log_probs) \
               - (1 - self.alpha) * ((1 - targets) * (probs ** self.gamma) * log_probs_neg)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class CombinedBCEDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1.0):
        super(CombinedBCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        self.alpha = alpha  # Weight for BCE loss

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        dice_loss = self.dice(preds, targets)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        return self.bce(inputs, targets)