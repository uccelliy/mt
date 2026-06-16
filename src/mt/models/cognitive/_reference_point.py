"""Decision-updated reference point model."""

from __future__ import annotations

import torch
import torch.nn as nn

from mt.models.common._base import FormulaOnlyCognitiveModel


def _prospect_decision_weights(
    probabilities: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return torch.sigmoid(a) + torch.sigmoid(b) * probabilities


def _prospect_utilities(
    values: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    e: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    positive = torch.sigmoid(c) * torch.pow(values.clamp_min(0), torch.sigmoid(d))
    scaled_losses = (-torch.sigmoid(f) * values).clamp_min(0)
    negative = -torch.sigmoid(e) * torch.pow(scaled_losses, torch.sigmoid(g))
    return torch.where(values >= 0, positive, negative)


def reference_point_logits(
    win_values: torch.Tensor,
    loss_values: torch.Tensor,
    win_probabilities: torch.Tensor,
    loss_probabilities: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    e: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    *,
    use_prospect_transforms: bool = True,
) -> torch.Tensor:
    if use_prospect_transforms:
        weighted_win = _prospect_decision_weights(
            win_probabilities.float(),
            a,
            b,
        ) * _prospect_utilities(win_values.float(), c, d, e, f, g)
        weighted_loss = _prospect_decision_weights(
            loss_probabilities.float(),
            a,
            b,
        ) * _prospect_utilities(loss_values.float(), c, d, e, f, g)
    else:
        weighted_win = win_values.float() * win_probabilities.float()
        weighted_loss = loss_values.float() * loss_probabilities.float()

    sample = h * (weighted_win + weighted_loss) + i
    stop = torch.zeros_like(sample) + j
    return torch.stack([stop, sample], dim=-1)


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
