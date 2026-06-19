from __future__ import annotations

import torch

from mt.models.cognitive._dual_systems import dual_systems_logits


def _two_step_tensors(num_participants: int = 1):
    choices = torch.tensor([[[0, 0], [1, 1], [0, 0]]]).repeat(num_participants, 1, 1)
    current_states = torch.tensor([[[0, 0], [0, 1], [0, 0]]]).repeat(num_participants, 1, 1)
    rewards = torch.tensor([[[0.0, 1.0], [0.0, -1.0], [0.0, 1.0]]]).repeat(num_participants, 1, 1)
    return choices, current_states, rewards


def test_dual_systems_logits_preserve_parameter_gradients() -> None:
    choices, current_states, rewards = _two_step_tensors()
    parameters = [torch.tensor(0.0, requires_grad=True) for _ in range(5)]

    logits = dual_systems_logits(
        choices,
        current_states,
        rewards,
        tau=parameters[0],
        alpha=parameters[1],
        lambd=parameters[2],
        stickiness=parameters[3],
        value_beta=parameters[4],
    )
    logits.sum().backward()

    assert logits.shape == (1, 3, 2, 2)
    assert all(parameter.grad is not None for parameter in parameters)


def test_participants_have_independent_stickiness_history() -> None:
    choices, current_states, rewards = _two_step_tensors(num_participants=2)

    logits = dual_systems_logits(
        choices,
        current_states,
        rewards,
        tau=torch.tensor(0.0),
        alpha=torch.tensor(0.0),
        lambd=torch.tensor(0.0),
        stickiness=torch.tensor(1.0),
        value_beta=torch.tensor(1.0),
    )

    torch.testing.assert_close(logits[0, 0, 0], torch.zeros(2))
    torch.testing.assert_close(logits[1, 0, 0], torch.zeros(2))
