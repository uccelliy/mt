"""Generalized context model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mt.models.common._base import BaseCognitiveModel
from mt.models.common._preprocessing import preprocess_generalized_context_data


def stack_feature_columns(data, feature_columns) -> torch.Tensor:
    """Stack separate feature tensors along the final feature dimension."""

    return torch.stack([data[column].float() for column in feature_columns], dim=-1)


def generalized_context_logits(
    features: torch.Tensor,
    memory_labels: torch.Tensor,
    *,
    num_classes: int,
    beta: torch.Tensor,
    ignore_index: int = -100,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute generalized context model logits."""

    features = features.float()
    distances = torch.cdist(features, features, p=2)
    similarities = torch.exp(-distances)

    num_trials = features.shape[1]
    previous_trial_mask = torch.tril(
        torch.ones(num_trials, num_trials, dtype=torch.bool, device=features.device),
        diagonal=-1,
    )
    valid_memory_mask = memory_labels != ignore_index
    similarities = similarities.masked_fill(
        ~(previous_trial_mask[None, :, :] & valid_memory_mask[:, None, :]),
        0.0,
    )

    safe_labels = memory_labels.long().clamp(0, num_classes - 1)
    labels = F.one_hot(safe_labels, num_classes=num_classes).to(similarities.dtype)
    evidence = similarities @ labels
    return torch.exp(beta) * torch.log(evidence + eps)


class GeneralizedContextModel(BaseCognitiveModel):
    def __init__(self, num_classes: int | None = None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = -100
        self.beta = nn.Parameter(0.01 * torch.randn([]))

    def preprocess_data(self, train_df, eval_df):
        return preprocess_generalized_context_data(
            train_df,
            eval_df,
            ignore_index=self.ignore_index,
        )

    def compute_logits(self, data):
        num_classes = self.num_classes if self.num_classes is not None else data["num_classes"]
        return generalized_context_logits(
            stack_feature_columns(data, data["feature_columns"]),
            data["ground_truth"],
            num_classes=int(num_classes),
            beta=self.beta,
            ignore_index=self.ignore_index,
        )
