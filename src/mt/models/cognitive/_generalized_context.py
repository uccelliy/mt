"""Generalized context model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mt.models.common._base import FormulaOnlyCognitiveModel


def generalized_context_logits(
    query_features: torch.Tensor,
    memory_features: torch.Tensor,
    memory_labels: torch.Tensor,
    *,
    num_classes: int,
    beta: torch.Tensor | float,
) -> torch.Tensor:
    """Compute generalized context model logits."""

    distances = torch.cdist(query_features.float(), memory_features.float(), p=2)
    similarities = torch.exp(-distances)
    labels = F.one_hot(memory_labels.long(), num_classes=num_classes).to(similarities.dtype)
    return beta * (similarities @ labels)


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
