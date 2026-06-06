"""Prospect-theory-style formulas."""

from __future__ import annotations

import torch


def prospect_decision_weights(probabilities, a, b):
    return torch.sigmoid(a) + torch.sigmoid(b) * probabilities


def prospect_utilities(values, c, d, e, f, g):
    positive = torch.sigmoid(c) * torch.pow(values.clamp_min(0), torch.sigmoid(d))
    scaled_losses = (-torch.sigmoid(f) * values).clamp_min(0)
    negative = -torch.sigmoid(e) * torch.pow(scaled_losses, torch.sigmoid(g))
    return torch.where(values >= 0, positive, negative)


def prospect_theory_logits(probabilities, values, beta, a, b, c, d, e, f, g):
    weights = prospect_decision_weights(probabilities.float(), a, b)
    utilities = prospect_utilities(values.float(), c, d, e, f, g)
    subjective_value = (weights * utilities).sum(dim=-1)
    return torch.exp(beta) * subjective_value


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
