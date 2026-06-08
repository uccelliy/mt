"""Linear model formulas."""

from __future__ import annotations

import torch


def online_linear_weights(features, rewards, alpha, initial_weight):
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


def multiple_cue_judgment_logits(features, rewards, option_values, alpha, beta, gamma, initial_weight):
    weights = online_linear_weights(features.float(), rewards.float(), alpha, initial_weight)
    prediction = (weights * features.float()).sum(dim=-1)
    return beta * torch.square(prediction[..., None] - option_values.float()) + gamma


def gardening_logits(features, rewards, alpha, beta, initial_weight):
    weights = online_linear_weights(features.float(), rewards.float(), alpha, initial_weight)
    value = (weights * features.float()).sum(dim=-1)
    accept = beta * value
    reject = torch.zeros_like(accept)
    return torch.stack([reject, accept], dim=-1)
