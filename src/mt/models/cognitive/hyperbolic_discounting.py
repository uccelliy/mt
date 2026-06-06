"""Hyperbolic discounting model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mt.models.cognitive.base import FormulaOnlyCognitiveModel
from mt.models.cognitive.formulas.discounting import hyperbolic_discounting_logits


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
