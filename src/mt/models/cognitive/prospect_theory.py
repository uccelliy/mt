"""Prospect theory model."""

from __future__ import annotations

import torch
import torch.nn as nn

from mt.models.cognitive.formula_base import FormulaOnlyCognitiveModel


def prospect_decision_weights(probabilities, a, b):
    return torch.sigmoid(a) + torch.sigmoid(b) * probabilities


def prospect_utilities(values, c, d, e, f, g):
    positive = torch.sigmoid(c) * torch.pow(values.clamp_min(0), torch.sigmoid(d))
    scaled_losses = (-torch.sigmoid(f) * values).clamp_min(0)
    negative = -torch.sigmoid(e) * torch.pow(scaled_losses, torch.sigmoid(g))
    return torch.where(values >= 0, positive, negative)


def prospect_theory_logits(probabilities, values, beta, a, b, c, d, e, f, g):
    """Compute option logits from probabilities and outcome values.

    Shapes:
        probabilities: ``(..., num_options, num_outcomes)``
        values: ``(..., num_options, num_outcomes)``
        output: ``(..., num_options)``
    """

    weights = prospect_decision_weights(probabilities.float(), a, b)
    utilities = prospect_utilities(values.float(), c, d, e, f, g)
    subjective_value = (weights * utilities).sum(dim=-1)
    return torch.exp(beta) * subjective_value


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

    def forward(self, data):
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
