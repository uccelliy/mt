"""Online linear regression choice models."""

from __future__ import annotations

import torch
import torch.nn as nn

from mt.models.cognitive._base import FormulaOnlyCognitiveModel
from mt.models.cognitive._formulas._linear import gardening_logits, multiple_cue_judgment_logits


class OnlineLinearRegressionModel(FormulaOnlyCognitiveModel):
    config_keys = ("num_features", "mode")

    def __init__(self, num_features: int, mode: str = "multiple_cue"):
        super().__init__()
        if mode not in {"multiple_cue", "gardening"}:
            raise ValueError("mode must be 'multiple_cue' or 'gardening'")

        self.num_features = num_features
        self.mode = mode
        self.alpha = nn.Parameter(0.01 * torch.randn([]))
        self.beta = nn.Parameter(0.01 * torch.randn([]))
        self.gamma = nn.Parameter(0.01 * torch.randn([]))
        self.initial_weight = nn.Parameter(0.01 * torch.randn(num_features))

    def compute_logits(self, data):
        if self.mode == "gardening":
            return gardening_logits(
                data["features"],
                data["reward"],
                self.alpha,
                self.beta,
                self.initial_weight,
            )

        return multiple_cue_judgment_logits(
            data["features"],
            data["reward"],
            data["option_values"],
            self.alpha,
            self.beta,
            self.gamma,
            self.initial_weight,
        )
