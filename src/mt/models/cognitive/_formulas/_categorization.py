"""Categorization model formulas."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
