"""Prospect theory model."""

from __future__ import annotations

import torch
import torch.nn as nn

from mt.models.cognitive.base import FormulaOnlyCognitiveModel
from mt.models.cognitive.formulas.prospect import prospect_theory_logits


class ProspectTheoryModel(FormulaOnlyCognitiveModel):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(0.01 * torch.randn([]))
        self.a = nn.Parameter(0.01 * torch.randn([]))
        self.b = nn.Parameter(0.01 * torch.randn([]))
        self.c = nn.Parameter(0.01 * torch.randn([]))
        self.d = nn.Parameter(0.01 * torch.randn([]))
        self.e = nn.Parameter(0.01 * torch.randn([]))
        self.f = nn.Parameter(0.01 * torch.randn([]))
        self.g = nn.Parameter(0.01 * torch.randn([]))

    def compute_logits(self, data):
        return prospect_theory_logits(
            data["probabilities"],
            data["values"],
            self.beta,
            self.a,
            self.b,
            self.c,
            self.d,
            self.e,
            self.f,
            self.g,
        )
