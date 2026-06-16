"""Online linear regression choice models."""

from __future__ import annotations

import torch
import torch.nn as nn

from mt.models.cognitive._base import FormulaOnlyCognitiveModel


def online_linear_weights(
    features: torch.Tensor,
    rewards: torch.Tensor,
    alpha: torch.Tensor,
    initial_weight: torch.Tensor,
) -> torch.Tensor:
    num_tasks, num_trials, num_features = features.shape
    weights = features.new_zeros((num_tasks, num_trials, num_features))
    weights[:, 0, :] = initial_weight

    for t in range(num_trials - 1):
        weights[:, t + 1, :] = weights[:, t, :]
        prediction = (weights[:, t, :] * features[:, t, :]).sum(dim=-1)
        error = rewards[:, t].float() - prediction
        error = torch.nan_to_num(error, nan=0.0)
        weights[:, t + 1, :] = weights[:, t, :] + alpha * error[:, None] * features[:, t, :]

    return weights


def multiple_cue_judgment_logits(
    features: torch.Tensor,
    rewards: torch.Tensor,
    option_values: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    initial_weight: torch.Tensor,
) -> torch.Tensor:
    weights = online_linear_weights(features.float(), rewards.float(), alpha, initial_weight)
    prediction = (weights * features.float()).sum(dim=-1)
    return beta * torch.square(prediction[..., None] - option_values.float()) + gamma


def gardening_logits(
    features: torch.Tensor,
    rewards: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_weight: torch.Tensor,
) -> torch.Tensor:
    weights = online_linear_weights(features.float(), rewards.float(), alpha, initial_weight)
    value = (weights * features.float()).sum(dim=-1)
    accept = beta * value
    reject = torch.zeros_like(accept)
    return torch.stack([reject, accept], dim=-1)


class OnlineLinearRegressionModel(FormulaOnlyCognitiveModel):
    config_keys = ("num_features", "mode")

    def __init__(self, num_features: int, mode: str = "multiple_cue"):
        super().__init__()
        if mode not in {"multiple_cue", "gardening"}:
            raise ValueError("mode must be 'multiple_cue' or 'gardening'")

        self.num_features = num_features
        self.mode = mode
        self.alpha = nn.Parameter(0.01 * torch.randn([]))
        self.beta = nn.Parameter(0.01 * torch.randn([]))
        self.gamma = nn.Parameter(0.01 * torch.randn([]))
        self.initial_weight = nn.Parameter(0.01 * torch.randn(num_features))

    def compute_logits(self, data):
        if self.mode == "gardening":
            return gardening_logits(
                data["features"],
                data["reward"],
                self.alpha,
                self.beta,
                self.initial_weight,
            )

        return multiple_cue_judgment_logits(
            data["features"],
            data["reward"],
            data["option_values"],
            self.alpha,
            self.beta,
            self.gamma,
            self.initial_weight,
        )
