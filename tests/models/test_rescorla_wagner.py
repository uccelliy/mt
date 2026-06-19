from __future__ import annotations

import torch

from mt.models.cognitive._rescorla_wagner import RescorlaWagnerModel, value_updating


def test_value_updating_uses_separate_learning_rates() -> None:
    values = value_updating(
        torch.tensor([[0, 1, 0]]),
        torch.tensor([[1.0, -1.0, float("nan")]]),
        num_options=2,
        alpha_plus=torch.tensor(0.0),
        alpha_minus=torch.tensor(0.0),
        initial_value=torch.tensor(0.0),
    )

    expected = torch.tensor([[[0.0, 0.0], [0.5, 0.0], [0.5, -0.5]]])
    torch.testing.assert_close(values, expected)


def test_value_updating_preserves_parameter_gradients() -> None:
    alpha_plus = torch.tensor(0.0, requires_grad=True)
    alpha_minus = torch.tensor(0.0, requires_grad=True)
    initial_value = torch.tensor(0.0, requires_grad=True)

    values = value_updating(
        torch.tensor([[0, 1, 0]]),
        torch.tensor([[1.0, -1.0, float("nan")]]),
        num_options=2,
        alpha_plus=alpha_plus,
        alpha_minus=alpha_minus,
        initial_value=initial_value,
    )
    values[:, -1].sum().backward()

    assert alpha_plus.grad is not None
    assert alpha_minus.grad is not None
    assert initial_value.grad is not None


def test_rescorla_wagner_parameters_live_on_model() -> None:
    parameter_names = dict(RescorlaWagnerModel().named_parameters())

    assert "alpha_plus" in parameter_names
    assert "alpha_minus" in parameter_names
    assert "initial_value" in parameter_names
    assert not any(name.startswith("value_updating.") for name in parameter_names)


def test_rescorla_wagner_loads_legacy_parameter_names() -> None:
    model = RescorlaWagnerModel()
    legacy_state = model.state_dict()
    legacy_state["value_updating.alpha_plus"] = legacy_state.pop("alpha_plus")
    legacy_state["value_updating.alpha_minus"] = legacy_state.pop("alpha_minus")
    legacy_state["value_updating.initial_values"] = legacy_state.pop("initial_value")

    RescorlaWagnerModel().load_state_dict(legacy_state)
