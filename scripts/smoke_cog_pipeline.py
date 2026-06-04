"""Smoke test for the cognitive-model dataframe pipeline.

Run from the repo root:

    python scripts/smoke_cog_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch
import torch.nn.functional as F

from mt.data.loading import load_dataframe
from mt.data.splitting import split_data_kfold
from mt.models.cog_model.rescola_wagner_model import RescorlaWagnerModel


def make_bandit_df(*, include_forced: bool) -> pd.DataFrame:
    rows = []
    for participant in range(4):
        for trial in range(6):
            choice = (participant + trial) % 3
            row = {
                "participant": participant,
                "task": "synthetic_bandit",
                "trial": trial,
                "choice": choice,
                "reward": float(choice == (trial % 3)),
            }
            if include_forced:
                row["forced"] = trial == 0
            rows.append(row)
    return pd.DataFrame(rows)


def run_rescorla_wagner_smoke(*, include_forced: bool) -> None:
    model = RescorlaWagnerModel(num_options=3)
    columns = list(model.required_columns)
    if include_forced:
        columns = columns + ["forced"]

    df = load_dataframe(make_bandit_df(include_forced=include_forced), columns=columns)
    train_df, eval_df = next(split_data_kfold(df, num_splits=2))
    train_data, eval_data = model.preprocess_data(train_df, eval_df)

    required_keys = {"choice", "reward", "choice_for_updating"}
    missing_keys = required_keys - set(train_data)
    assert not missing_keys, f"Missing train_data keys: {sorted(missing_keys)}"

    train_logits = model(train_data)
    eval_logits = model(eval_data)

    expected_train_shape = (*train_data["choice"].shape, model.num_options)
    expected_eval_shape = (*eval_data["choice"].shape, model.num_options)
    assert train_logits.shape == expected_train_shape
    assert eval_logits.shape == expected_eval_shape

    train_loss = F.cross_entropy(
        train_logits.flatten(0, -2),
        train_data["choice"].flatten().long(),
        ignore_index=model.ignore_index,
    )
    assert torch.isfinite(train_loss)

    if include_forced:
        assert "forced" in train_data
        assert (train_data["choice"] == model.ignore_index).any()


def main() -> None:
    run_rescorla_wagner_smoke(include_forced=False)
    run_rescorla_wagner_smoke(include_forced=True)
    print("cognitive pipeline smoke test passed")


if __name__ == "__main__":
    main()
