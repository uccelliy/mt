"""Reinforcement-learning formulas."""

from __future__ import annotations

import torch


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
