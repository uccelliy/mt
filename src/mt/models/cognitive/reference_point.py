"""Decision-updated reference point model."""

from __future__ import annotations

import torch
import torch.nn as nn

from mt.models.cognitive.base import FormulaOnlyCognitiveModel
from mt.models.cognitive.formulas.prospect import reference_point_logits


class DecisionUpdatedReferencePointModel(FormulaOnlyCognitiveModel):
    config_keys = ("use_prospect_transforms",)

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

    def compute_logits(self, data):
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
