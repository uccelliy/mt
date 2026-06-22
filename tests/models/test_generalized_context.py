from __future__ import annotations

import torch

from mt.models.cognitive._generalized_context import (
    GeneralizedContextModel,
    stack_feature_columns,
)


def test_stack_feature_columns_adds_a_final_feature_dimension() -> None:
    data = {
        "x1": torch.tensor([[1.0, 2.0]]),
        "x2": torch.tensor([[3.0, 4.0]]),
    }

    features = stack_feature_columns(data, ("x1", "x2"))

    expected = torch.tensor([[[1.0, 3.0], [2.0, 4.0]]])
    torch.testing.assert_close(features, expected)
    assert set(data) == {"x1", "x2"}


def test_generalized_context_model_stacks_preprocessed_feature_columns() -> None:
    model = GeneralizedContextModel(num_classes=2)
    data = {
        "choice": torch.tensor([[0, 1]]),
        "ground_truth": torch.tensor([[0, 1]]),
        "x1": torch.tensor([[1.0, 2.0]]),
        "x2": torch.tensor([[3.0, 4.0]]),
        "feature_columns": ("x1", "x2"),
        "num_classes": 2,
    }

    logits = model.compute_logits(data)

    assert logits.shape == (1, 2, 2)
