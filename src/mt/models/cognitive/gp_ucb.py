"""GP-UCB choice rule."""

from __future__ import annotations

import torch
import torch.nn as nn

from mt.models.cognitive.formula_base import FormulaOnlyCognitiveModel


def gp_ucb_logits(means, stds, beta, gamma):
    """Compute logits ``beta * (mean + exp(gamma) * std)``."""

    return beta * (means.float() + torch.exp(gamma) * stds.float())


class GPUCBModel(FormulaOnlyCognitiveModel):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(0.01 * torch.randn([]))
        self.gamma = nn.Parameter(0.01 * torch.randn([]))

    def forward(self, data):
        return gp_ucb_logits(data["means"], data["stds"], self.beta, self.gamma)
