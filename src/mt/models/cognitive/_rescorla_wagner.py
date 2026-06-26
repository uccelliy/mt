"""Rescorla-Wagner model."""

from __future__ import annotations

import torch
import torch.nn as nn

from mt.models.common._base import BaseCognitiveModel
from mt.models.common._preprocessing import preprocess_rescorla_wagner_data


def previous_choice_features(choices: torch.Tensor, num_options: int) -> torch.Tensor:
    """Indicator features for whether each option was chosen on the previous trial."""

    num_tasks = choices.shape[0]
    previous_choices_padding = choices.new_zeros((num_tasks, 1, num_options), dtype=torch.float32)
    previous_choices = torch.stack(
        [(choices[:, :-1] == option).float() for option in range(num_options)],
        dim=-1,
    )
    return torch.cat([previous_choices_padding, previous_choices], dim=1)


def cumulative_choice_features(choices: torch.Tensor, num_options: int) -> torch.Tensor:
    """Counts of how often each option has been chosen before each trial."""

    num_tasks = choices.shape[0]
    cumsum_choices_padding = choices.new_zeros((num_tasks, 1, num_options), dtype=torch.float32)
    cumsum_choices = torch.stack(
        [torch.cumsum((choices[:, :-1] == option).float(), dim=1) for option in range(num_options)],
        dim=-1,
    )
    return torch.cat([cumsum_choices_padding, cumsum_choices], dim=1)


def value_updating(
    choices: torch.Tensor,
    rewards: torch.Tensor,
    *,
    num_options: int,
    alpha_plus: torch.Tensor,
    alpha_minus: torch.Tensor,
    initial_value: torch.Tensor,
    max_initial_value: float = 100.0,
) -> torch.Tensor:
    """Update option values with separate learning rates for positive and negative errors."""

    choices = choices.long()
    rewards = rewards.float()
    num_tasks, num_trials = choices.shape
    row = torch.arange(num_tasks, device=choices.device)

    bounded_initial_value = max_initial_value * torch.tanh(initial_value)
    positive_learning_rate = torch.sigmoid(alpha_plus)
    negative_learning_rate = torch.sigmoid(alpha_minus)
    q_value_history = bounded_initial_value.expand(num_tasks, num_trials, num_options).clone()

    for trial in range(num_trials - 1):
        q_value_history[:, trial + 1] = q_value_history[:, trial]
        choice = choices[:, trial]
        prediction_error = rewards[:, trial] - q_value_history[row, trial, choice]
        prediction_error = torch.nan_to_num(prediction_error, nan=0.0)
        learning_rate = torch.where(
            prediction_error >= 0,
            positive_learning_rate,
            negative_learning_rate,
        )
        q_value_history[row, trial + 1, choice] = (
            q_value_history[row, trial, choice] + learning_rate * prediction_error
        )

    return q_value_history


def rescorla_wagner_logits(
    choices: torch.Tensor,
    choices_for_updating: torch.Tensor,
    rewards: torch.Tensor,
    *,
    num_options: int,
    alpha_plus: torch.Tensor,
    alpha_minus: torch.Tensor,
    initial_value: torch.Tensor,
    value_beta: torch.Tensor,
    stickiness_beta: torch.Tensor,
    information_beta: torch.Tensor,
    max_initial_value: float = 100.0,
) -> torch.Tensor:
    """Compute Rescorla-Wagner choice logits."""

    q_value_history = value_updating(
        choices_for_updating.long(),
        rewards.float(),
        num_options=num_options,
        alpha_plus=alpha_plus,
        alpha_minus=alpha_minus,
        initial_value=initial_value,
        max_initial_value=max_initial_value,
    )
    choices = choices.long()
    information_logits = information_beta * cumulative_choice_features(choices, num_options)
    stickiness_logits = stickiness_beta * previous_choice_features(choices, num_options)
    value_logits = value_beta * q_value_history
    return value_logits + stickiness_logits + information_logits


class RescorlaWagnerModel(BaseCognitiveModel):

    def __init__(self, num_options: int = 3):
        super().__init__()

        self.num_options = num_options
        self.ignore_index = -100
        self.max_initial_value = 100.0

        self.alpha_plus = nn.Parameter(0.01 * torch.randn([]))
        self.alpha_minus = nn.Parameter(0.01 * torch.randn([]))
        self.initial_value = nn.Parameter(0.01 * torch.randn([]))
        self.value_beta = nn.Parameter(0.01 * torch.randn([]))
        self.stickiness_beta = nn.Parameter(0.01 * torch.randn([]))
        self.information_beta = nn.Parameter(0.01 * torch.randn([]))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        legacy_names = {
            "value_updating.alpha_plus": "alpha_plus",
            "value_updating.alpha_minus": "alpha_minus",
            "value_updating.initial_values": "initial_value",
        }
        for legacy_name, current_name in legacy_names.items():
            legacy_key = f"{prefix}{legacy_name}"
            current_key = f"{prefix}{current_name}"
            if legacy_key in state_dict and current_key not in state_dict:
                state_dict[current_key] = state_dict.pop(legacy_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def preprocess_data(self, train_df, eval_df):
        """
        Preprocesses data into pytorch format.

        Parameter
        ---------
        train_df : pandas dataframe.
        eval_df : pandas dataframe.

        Returns
        -------
        dict, dict
        Dictionaries with tensors named 'choice' and 'reward' of shape (N, T).
        """

        return preprocess_rescorla_wagner_data(
            train_df,
            eval_df,
            ignore_index=self.ignore_index,
        )

    def compute_logits(self, data):
        """
        Model with Rescorla-Wagner-based learning, stickiness and information bonus.

        Parameter
        ---------
        data : dictionary with tensors named 'choice' and 'reward' of shape (N, T).

        Returns
        -------
        tensor of shape (N, T, self.num_options)
        Tensor filled with logits for all options.
        """

        return rescorla_wagner_logits(
            data["choice"],
            data["choice_for_updating"],
            data["reward"],
            num_options=self.num_options,
            alpha_plus=self.alpha_plus,
            alpha_minus=self.alpha_minus,
            initial_value=self.initial_value,
            value_beta=self.value_beta,
            stickiness_beta=self.stickiness_beta,
            information_beta=self.information_beta,
            max_initial_value=self.max_initial_value,
        )
