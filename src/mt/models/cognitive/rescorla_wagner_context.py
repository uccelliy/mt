"""Rescorla-Wagner model with context."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mt.models.cognitive.formula_base import FormulaOnlyCognitiveModel


def rescorla_wagner_context_values(
    choices: torch.Tensor,
    rewards: torch.Tensor,
    contexts: torch.Tensor,
    *,
    num_contexts: int,
    num_options: int,
    alpha: torch.Tensor,
    initial_value: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute context-specific RW values.

    Shapes:
        choices, rewards, contexts: ``(N, T)``
        output: ``(N, T, num_contexts, num_options)``
    """

    num_tasks, num_trials = choices.shape
    values = choices.new_zeros(
        (num_tasks, num_trials, num_contexts, num_options),
        dtype=torch.float32,
    )
    values[:, 0, :, :] = initial_value

    row = torch.arange(num_tasks, device=choices.device)
    for t in range(num_trials - 1):
        values[:, t + 1] = values[:, t]

        choice_t = choices[:, t].long()
        context_t = contexts[:, t].long()
        valid = (
            (choice_t != ignore_index)
            & (context_t >= 0)
            & (context_t < num_contexts)
            & (choice_t >= 0)
            & (choice_t < num_options)
            & (~torch.isnan(rewards[:, t]))
        )
        if not valid.any():
            continue

        pred = values[row[valid], t, context_t[valid], choice_t[valid]]
        prediction_error = rewards[valid, t].float() - pred
        values[row[valid], t + 1, context_t[valid], choice_t[valid]] = pred + alpha * prediction_error

    return values


def rescorla_wagner_context_logits(values, contexts, beta):
    row = torch.arange(values.shape[0], device=values.device)[:, None]
    trial = torch.arange(values.shape[1], device=values.device)[None, :]
    safe_contexts = contexts.long().clamp(0, values.shape[2] - 1)
    return beta * values[row, trial, safe_contexts]


class RescorlaWagnerContextModel(FormulaOnlyCognitiveModel):
    def __init__(self, num_contexts: int, num_options: int, ignore_index: int = -100):
        super().__init__()
        self.num_contexts = num_contexts
        self.num_options = num_options
        self.ignore_index = ignore_index
        self.alpha = nn.Parameter(0.01 * torch.randn([]))
        self.beta = nn.Parameter(0.01 * torch.randn([]))
        self.initial_value = nn.Parameter(0.01 * torch.randn([]))

    def forward(self, data):
        values = rescorla_wagner_context_values(
            data["choice"].long(),
            data["reward"].float(),
            data["context"].long(),
            num_contexts=self.num_contexts,
            num_options=self.num_options,
            alpha=torch.sigmoid(self.alpha),
            initial_value=self.initial_value,
            ignore_index=self.ignore_index,
        )
        return rescorla_wagner_context_logits(values, data["context"], self.beta)
