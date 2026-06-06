"""Reinforcement-learning formulas."""

from __future__ import annotations

import torch


def previous_choice_features(choices: torch.Tensor, num_options: int) -> torch.Tensor:
    """Indicator features for whether each option was chosen on the previous trial."""

    num_tasks = choices.shape[0]
    previous_choices_0 = choices.new_zeros((num_tasks, 1, num_options), dtype=torch.float32)
    previous_choices_1 = torch.stack(
        [(choices[:, :-1] == option).float() for option in range(num_options)],
        dim=-1,
    )
    return torch.cat([previous_choices_0, previous_choices_1], dim=1)


def cumulative_choice_features(choices: torch.Tensor, num_options: int) -> torch.Tensor:
    """Counts of how often each option has been chosen before each trial."""

    num_tasks = choices.shape[0]
    cumsum_choices_0 = choices.new_zeros((num_tasks, 1, num_options), dtype=torch.float32)
    cumsum_choices_1 = torch.stack(
        [
            torch.cumsum((choices[:, :-1] == option).float(), dim=1)
            for option in range(num_options)
        ],
        dim=-1,
    )
    return torch.cat([cumsum_choices_0, cumsum_choices_1], dim=1)


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
