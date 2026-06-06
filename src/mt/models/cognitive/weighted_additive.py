"""Weighted-additive model."""

from __future__ import annotations

import torch
import torch.nn as nn

from mt.models.cognitive.base import FormulaOnlyCognitiveModel
from mt.models.cognitive.formulas.choice import weighted_additive_logits


class WeightedAdditiveModel(FormulaOnlyCognitiveModel):
    config_keys = ("num_features",)

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.weights = nn.Parameter(0.01 * torch.randn(num_features))

    def compute_logits(self, data):
        return weighted_additive_logits(data["option_features"], self.weights)
