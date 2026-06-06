"""Generalized context model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mt.models.cognitive.formula_base import FormulaOnlyCognitiveModel


def generalized_context_logits(
    query_features: torch.Tensor,
    memory_features: torch.Tensor,
    memory_labels: torch.Tensor,
    *,
    num_classes: int,
    beta: torch.Tensor | float,
) -> torch.Tensor:
    """Compute GCM logits.

    Shapes:
        query_features: ``(N, D)``
        memory_features: ``(M, D)``
        memory_labels: ``(M,)`` integer class labels
        output: ``(N, num_classes)``
    """

    distances = torch.cdist(query_features.float(), memory_features.float(), p=2)
    similarities = torch.exp(-distances)
    labels = F.one_hot(memory_labels.long(), num_classes=num_classes).to(similarities.dtype)
    return beta * (similarities @ labels)


class GeneralizedContextModel(FormulaOnlyCognitiveModel):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.beta = nn.Parameter(0.01 * torch.randn([]))

    def forward(self, data):
        return generalized_context_logits(
            data["query_features"],
            data["memory_features"],
            data["memory_labels"],
            num_classes=self.num_classes,
            beta=self.beta,
        )
