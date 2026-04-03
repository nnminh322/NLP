"""Loss functions for CACL training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Margin-based triplet loss.
    L = max(0, m − s(Q,C+) + s(Q,C−))
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        loss = F.relu(self.margin - pos_scores + neg_scores)
        return loss.mean()


class ConstraintViolationLoss(nn.Module):
    """
    Penalise high scores on constraint-violating negatives.
    L = −(1/N) Σ 1[violates] · log σ(−s(Q,C−))

    When s(Q,C-) is high and doc violates constraints:
        σ(-s) → 0, log σ(-s) → -∞, loss → +∞ (strong penalty)
    When s(Q,C-) is low:
        σ(-s) → 1, log σ(-s) → 0, loss → 0 (no penalty)
    """

    def forward(
        self,
        neg_scores: torch.Tensor,
        violates_mask: torch.Tensor,
    ) -> torch.Tensor:
        # log σ(-s) = -softplus(s) — numerically stable
        log_sigma_neg_s = -F.softplus(neg_scores)
        loss = -(violates_mask * log_sigma_neg_s).mean()
        return loss


class CACLLoss(nn.Module):
    """
    Combined CACL loss: L = L_triplet + λ · L_constraint
    """

    def __init__(self, margin: float = 0.2, lambda_constraint: float = 0.5):
        super().__init__()
        self.triplet = TripletLoss(margin=margin)
        self.constraint = ConstraintViolationLoss()
        self.lambda_constraint = lambda_constraint

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        violates_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        L_triplet = self.triplet(pos_scores, neg_scores)
        L_constraint = self.constraint(neg_scores, violates_mask)
        L_total = L_triplet + self.lambda_constraint * L_constraint
        return {
            "total": L_total,
            "triplet": L_triplet,
            "constraint": L_constraint,
        }
