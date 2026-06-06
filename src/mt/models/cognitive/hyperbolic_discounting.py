"""Hyperbolic discounting model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mt.models.cognitive.formula_base import FormulaOnlyCognitiveModel


def hyperbolic_discounting_logits(rewards, delays, beta, discount_rate):
    """Compute logits ``beta * reward / (1 + discount_rate * delay)``."""

    return beta * rewards.float() / (1 + discount_rate * delays.float())


class HyperbolicDiscountingModel(FormulaOnlyCognitiveModel):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(0.01 * torch.randn([]))
        self.discount_rate = nn.Parameter(0.01 * torch.randn([]))

    def forward(self, data):
        return hyperbolic_discounting_logits(
            data["rewards"],
            data["delays"],
            self.beta,
            F.softplus(self.discount_rate),
        )
