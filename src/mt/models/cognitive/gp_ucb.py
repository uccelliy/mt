"""GP-UCB choice rule."""

from __future__ import annotations

import torch
import torch.nn as nn

from mt.models.cognitive.base import FormulaOnlyCognitiveModel
from mt.models.cognitive.formulas.choice import gp_ucb_logits


class GPUCBModel(FormulaOnlyCognitiveModel):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(0.01 * torch.randn([]))
        self.gamma = nn.Parameter(0.01 * torch.randn([]))

    def compute_logits(self, data):
        return gp_ucb_logits(data["means"], data["stds"], self.beta, self.gamma)
