import torch
import torch.nn as nn
import torch.nn.functional as F
from mt.models.common._base import BaseCognitiveModel
from mt.models.common._preprocessing import preprocess_rescorla_wagner_data


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
        [torch.cumsum((choices[:, :-1] == option).float(), dim=1) for option in range(num_options)],
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
        values[row[valid], t + 1, context_t[valid], choice_t[valid]] = (
            pred + alpha * prediction_error
        )

    return values


def rescorla_wagner_context_logits(
    values: torch.Tensor,
    contexts: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    row = torch.arange(values.shape[0], device=values.device)[:, None]
    trial = torch.arange(values.shape[1], device=values.device)[None, :]
    safe_contexts = contexts.long().clamp(0, values.shape[2] - 1)
    return beta * values[row, trial, safe_contexts]


class RescorlaWagnerModel(BaseCognitiveModel):
    config_keys = ("num_options",)

    def __init__(self, num_options=3):
        super().__init__()

        self.num_options = num_options
        self.ignore_index = -100

        self.value_updating = TabularRescorlaWagnerPlusMinusValueUpdating(num_options)

        self.value_beta = nn.Parameter(0.01 * torch.randn([]))
        self.stickiness_beta = nn.Parameter(0.01 * torch.randn([]))
        self.information_beta = nn.Parameter(0.01 * torch.randn([]))

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

        values = self.value_updating(data["choice_for_updating"].long(), data["reward"].float())

        choices = data["choice"].long()
        information_logits = self.information_beta * cumulative_choice_features(
            choices,
            self.num_options,
        )
        stickiness_logits = self.stickiness_beta * previous_choice_features(
            choices,
            self.num_options,
        )
        value_logits = self.value_beta * values
        return value_logits + stickiness_logits + information_logits


class TabularRescorlaWagnerPlusMinusValueUpdating(nn.Module):
    def __init__(self, num_options, max_initial_values=100, ignore_index=-100):
        super().__init__()

        self.num_options = num_options
        self.max_initial_values = max_initial_values

        self.alpha_plus = nn.Parameter(0.01 * torch.randn([]))
        self.alpha_minus = nn.Parameter(0.01 * torch.randn([]))
        self.initial_values = nn.Parameter(0.01 * torch.randn([]))

        self.ignore_index = ignore_index

    def forward(self, choices, rewards):
        """
        Performs Rescorla-Wagner updating with separate learning rates for positive and negative prediction errors for the given choices and rewards.

        Parameter
        ---------
        choices : tensor of shape (N, T).
        rewards : tensor of shape (N, T).

        Returns
        -------
        tensor of shape (N, T, self.num_options)
        Tensor filled with estimated values for all options.
        """

        num_tasks = choices.shape[0]
        num_trials = choices.shape[1]

        initial_values = self.max_initial_values * F.tanh(self.initial_values)
        alpha_plus = F.sigmoid(self.alpha_plus)
        alpha_minus = F.sigmoid(self.alpha_minus)

        values = torch.ones(list(choices.shape) + [self.num_options]) * initial_values

        for t in range(num_trials - 1):
            # copy over everything
            values[:, t + 1, :] = values[:, t, :]
            # compute prediction errors
            prediction_error = rewards[:, t] - values[torch.arange(num_tasks), t, choices[:, t]]
            # zero-out prediction errors for missing trial
            prediction_error[torch.isnan(rewards[:, t])] = 0
            # update values for selected actions
            values[torch.arange(num_tasks), t + 1, choices[:, t]] = (
                values[torch.arange(num_tasks), t, choices[:, t]]
                + (alpha_plus * prediction_error * (prediction_error >= 0).float())
                + (alpha_minus * prediction_error * (prediction_error < 0).float())
            )

        return values
