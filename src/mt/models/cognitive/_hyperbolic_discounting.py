"""Hyperbolic discounting model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mt.models.cognitive._base import FormulaOnlyCognitiveModel


def hyperbolic_discounting_logits(
    rewards: torch.Tensor,
    delays: torch.Tensor,
    beta: torch.Tensor,
    discount_rate: torch.Tensor,
) -> torch.Tensor:
    return beta * rewards.float() / (1 + discount_rate * delays.float())


class HyperbolicDiscountingModel(FormulaOnlyCognitiveModel):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(0.01 * torch.randn([]))
        self.discount_rate = nn.Parameter(0.01 * torch.randn([]))

    def compute_logits(self, data):
        return hyperbolic_discounting_logits(
            data["rewards"],
            data["delays"],
            self.beta,
            F.softplus(self.discount_rate),
        )
