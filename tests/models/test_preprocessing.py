from __future__ import annotations

import pandas as pd
import pytest

from mt.models.common._preprocessing import preprocess_generalized_context_data


def _gcm_dataframe(participant: int, choices: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant": [participant] * len(choices),
            "task": [0] * len(choices),
            "trial": list(range(len(choices))),
            "choice": choices,
            "ground_truth": choices,
            "x1": [float(index) for index in range(len(choices))],
        }
    )


def test_generalized_context_class_mapping_is_learned_from_training_data() -> None:
    train_df = _gcm_dataframe(1, [10, 20])
    eval_df = _gcm_dataframe(2, [20, 10])

    train_data, eval_data = preprocess_generalized_context_data(train_df, eval_df)

    assert train_data["num_classes"] == 2
    assert eval_data["num_classes"] == 2
    assert train_data["choice"].tolist() == [[0, 1]]
    assert eval_data["choice"].tolist() == [[1, 0]]


def test_generalized_context_rejects_class_seen_only_in_evaluation_data() -> None:
    train_df = _gcm_dataframe(1, [10, 20])
    eval_df = _gcm_dataframe(2, [10, 30])

    with pytest.raises(ValueError, match="not observed in training data"):
        preprocess_generalized_context_data(train_df, eval_df)
