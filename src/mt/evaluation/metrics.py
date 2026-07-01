"""Evaluation metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def choice_nll(logits: torch.Tensor, choices: torch.Tensor) -> torch.Tensor:
    """Compute mean negative log-likelihood for choice logits."""

    return F.cross_entropy(logits.flatten(0, -2), choices.flatten().long())
