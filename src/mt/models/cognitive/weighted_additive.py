"""Weighted-additive model."""

from __future__ import annotations

import torch
import torch.nn as nn

from mt.models.cognitive.formula_base import FormulaOnlyCognitiveModel


def weighted_additive_logits(option_features, weights):
    """Compute logits ``x_i @ w`` for each option."""

    return torch.matmul(option_features.float(), weights.float())


class WeightedAdditiveModel(FormulaOnlyCognitiveModel):
    def __init__(self, num_features: int):
        super().__init__()
        self.weights = nn.Parameter(0.01 * torch.randn(num_features))

    def forward(self, data):
        return weighted_additive_logits(data["option_features"], self.weights)
