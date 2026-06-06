"""Decision-updated reference point model."""

from __future__ import annotations

import torch
import torch.nn as nn

from mt.models.cognitive.formula_base import FormulaOnlyCognitiveModel
from mt.models.cognitive.prospect_theory import prospect_decision_weights, prospect_utilities


def reference_point_logits(
    win_values,
    loss_values,
    win_probabilities,
    loss_probabilities,
    a,
    b,
    c,
    d,
    e,
    f,
    g,
    h,
    i,
    j,
    *,
    use_prospect_transforms: bool = True,
):
    """Compute sample/stop logits for the Columbia card task."""

    if use_prospect_transforms:
        weighted_win = prospect_decision_weights(win_probabilities.float(), a, b) * prospect_utilities(
            win_values.float(), c, d, e, f, g
        )
        weighted_loss = prospect_decision_weights(loss_probabilities.float(), a, b) * prospect_utilities(
            loss_values.float(), c, d, e, f, g
        )
    else:
        weighted_win = win_values.float() * win_probabilities.float()
        weighted_loss = loss_values.float() * loss_probabilities.float()

    sample = h * (weighted_win + weighted_loss) + i
    stop = torch.zeros_like(sample) + j
    return torch.stack([stop, sample], dim=-1)


class DecisionUpdatedReferencePointModel(FormulaOnlyCognitiveModel):
    def __init__(self, *, use_prospect_transforms: bool = True):
        super().__init__()
        self.use_prospect_transforms = use_prospect_transforms
        self.a = nn.Parameter(0.01 * torch.randn([]))
        self.b = nn.Parameter(0.01 * torch.randn([]))
        self.c = nn.Parameter(0.01 * torch.randn([]))
        self.d = nn.Parameter(0.01 * torch.randn([]))
        self.e = nn.Parameter(0.01 * torch.randn([]))
        self.f = nn.Parameter(0.01 * torch.randn([]))
        self.g = nn.Parameter(0.01 * torch.randn([]))
        self.h = nn.Parameter(0.01 * torch.randn([]))
        self.i = nn.Parameter(0.01 * torch.randn([]))
        self.j = nn.Parameter(0.01 * torch.randn([]))

    def forward(self, data):
        return reference_point_logits(
            data["win_values"],
            data["loss_values"],
            data["win_probabilities"],
            data["loss_probabilities"],
            self.a,
            self.b,
            self.c,
            self.d,
            self.e,
            self.f,
            self.g,
            self.h,
            self.i,
            self.j,
            use_prospect_transforms=self.use_prospect_transforms,
        )
