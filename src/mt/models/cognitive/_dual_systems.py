"""Dual-systems model for two-step decision tasks."""

from __future__ import annotations

import torch
import torch.nn as nn

from mt.models.common._base import BaseCognitiveModel
from mt.models.common._preprocessing import preprocess_dual_system_data


def dual_systems_logits(
    choices: torch.Tensor,
    current_states: torch.Tensor,
    rewards: torch.Tensor,
    *,
    tau: torch.Tensor,
    alpha: torch.Tensor,
    lambd: torch.Tensor,
    stickiness: torch.Tensor,
    value_beta: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute two-step choice logits from model-based and model-free values."""

    model_based_weight = torch.sigmoid(tau)
    learning_rate = torch.sigmoid(alpha)
    eligibility_trace = torch.sigmoid(lambd)
    perseveration = torch.tanh(stickiness)

    first_actions = choices[:, :, 0].long()
    second_actions = choices[:, :, 1].long()
    second_states = current_states[:, :, 1].long()
    second_rewards = rewards[:, :, 1]

    transition_matrix = value_beta.new_tensor([[0.7, 0.3], [0.3, 0.7]])
    num_participants, num_trials = first_actions.shape
    participant_logits = []

    for participant in range(num_participants):
        model_free_values = value_beta.new_zeros((3, 2))
        previous_action = value_beta.new_zeros(2)
        trial_logits = []

        for trial in range(num_trials):
            best_second_stage_values = model_free_values[1:].max(dim=1).values
            model_based_values = transition_matrix @ best_second_stage_values
            combined_values = (
                model_based_weight * model_based_values
                + (1 - model_based_weight) * model_free_values[0]
            )

            first_action = first_actions[participant, trial]
            if first_action == ignore_index:
                trial_logits.append(value_beta.new_zeros((2, 2)))
                continue

            second_state = second_states[participant, trial]
            second_action = second_actions[participant, trial]
            trial_logits.append(
                torch.stack(
                    [
                        combined_values + previous_action * perseveration,
                        model_free_values[second_state + 1],
                    ]
                )
            )

            if second_action != ignore_index:
                first_stage_error = (
                    model_free_values[second_state + 1, second_action]
                    - model_free_values[0, first_action]
                )
                reward_error = (
                    second_rewards[participant, trial]
                    - model_free_values[second_state + 1, second_action]
                )
                first_stage_mask = value_beta.new_zeros((3, 2))
                first_stage_mask[0, first_action] = 1
                second_stage_mask = value_beta.new_zeros((3, 2))
                second_stage_mask[second_state + 1, second_action] = 1
                model_free_values = (
                    model_free_values
                    + first_stage_mask
                    * (
                        learning_rate * first_stage_error
                        + eligibility_trace * learning_rate * reward_error
                    )
                    + second_stage_mask * (learning_rate * reward_error)
                )

            previous_action = value_beta.new_zeros(2)
            previous_action[first_action] = 1

        participant_logits.append(torch.stack(trial_logits))

    return value_beta * torch.stack(participant_logits)


class DualSystemsModel(BaseCognitiveModel):
    def __init__(self):
        super().__init__()

        self.ignore_index = -100

        self.tau = nn.Parameter(torch.randn([]))
        self.alpha = nn.Parameter(torch.randn([]))
        self.lambd = nn.Parameter(torch.randn([]))
        self.stickiness = nn.Parameter(torch.randn([]))

        self.value_beta = nn.Parameter(0.01 * torch.randn([]))

    def preprocess_data(self, train_df, eval_df):
        return preprocess_dual_system_data(train_df, eval_df, ignore_index=self.ignore_index)

    def compute_logits(self, data):
        return dual_systems_logits(
            data["choice"],
            data["current_state"],
            data["reward"],
            tau=self.tau,
            alpha=self.alpha,
            lambd=self.lambd,
            stickiness=self.stickiness,
            value_beta=self.value_beta,
            ignore_index=self.ignore_index,
        )
