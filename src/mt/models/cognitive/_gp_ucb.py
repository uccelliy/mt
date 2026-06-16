"""GP-UCB choice rule."""

from __future__ import annotations

import torch
import torch.nn as nn

from mt.models.cognitive._base import FormulaOnlyCognitiveModel


def gp_ucb_logits(
    means: torch.Tensor,
    stds: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    return beta * (means.float() + torch.exp(gamma) * stds.float())


class GPUCBModel(FormulaOnlyCognitiveModel):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(0.01 * torch.randn([]))
        self.gamma = nn.Parameter(0.01 * torch.randn([]))

    def compute_logits(self, data):
        return gp_ucb_logits(data["means"], data["stds"], self.beta, self.gamma)
