"""Generalized context model."""

from __future__ import annotations

import torch
import torch.nn as nn

from mt.models.cognitive.base import FormulaOnlyCognitiveModel
from mt.models.cognitive.formulas.categorization import generalized_context_logits


class GeneralizedContextModel(FormulaOnlyCognitiveModel):
    config_keys = ("num_classes",)

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.beta = nn.Parameter(0.01 * torch.randn([]))

    def compute_logits(self, data):
        return generalized_context_logits(
            data["query_features"],
            data["memory_features"],
            data["memory_labels"],
            num_classes=self.num_classes,
            beta=self.beta,
        )
