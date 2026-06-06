"""Base helpers for formula-only cognitive models."""

from __future__ import annotations

import torch.nn as nn


class FormulaOnlyCognitiveModel(nn.Module):
    """Cognitive model with implemented equations but no dataframe contract yet."""

    required_columns: list[str] = []

    def preprocess_data(self, train_df, eval_df):
        raise NotImplementedError(
            f"{self.__class__.__name__} has formula code but no data contract yet."
        )
