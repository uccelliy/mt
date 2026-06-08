"""Rescorla-Wagner model with context."""

from __future__ import annotations

import torch
import torch.nn as nn

from mt.models.cognitive._base import FormulaOnlyCognitiveModel
from mt.models.cognitive._formulas._reinforcement_learning import (
    rescorla_wagner_context_logits,
    rescorla_wagner_context_values,
)


class RescorlaWagnerContextModel(FormulaOnlyCognitiveModel):
    config_keys = ("num_contexts", "num_options", "ignore_index")

    def __init__(self, num_contexts: int, num_options: int, ignore_index: int = -100):
        super().__init__()
        self.num_contexts = num_contexts
        self.num_options = num_options
        self.ignore_index = ignore_index
        self.alpha = nn.Parameter(0.01 * torch.randn([]))
        self.beta = nn.Parameter(0.01 * torch.randn([]))
        self.initial_value = nn.Parameter(0.01 * torch.randn([]))

    def compute_logits(self, data):
        values = rescorla_wagner_context_values(
            data["choice"].long(),
            data["reward"].float(),
            data["context"].long(),
            num_contexts=self.num_contexts,
            num_options=self.num_options,
            alpha=torch.sigmoid(self.alpha),
            initial_value=self.initial_value,
            ignore_index=self.ignore_index,
        )
        return rescorla_wagner_context_logits(values, data["context"], self.beta)
