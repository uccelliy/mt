"""Choice-rule formulas."""

from __future__ import annotations

import torch


def weighted_additive_logits(option_features, weights):
    return torch.matmul(option_features.float(), weights.float())


def odd_one_out_logits(option_embeddings):
    if option_embeddings.shape[-2] != 3:
        raise ValueError("odd_one_out_logits expects exactly three options")

    x0 = option_embeddings[..., 0, :]
    x1 = option_embeddings[..., 1, :]
    x2 = option_embeddings[..., 2, :]
    return torch.stack(
        [
            (x1 * x2).sum(dim=-1),
            (x0 * x2).sum(dim=-1),
            (x0 * x1).sum(dim=-1),
        ],
        dim=-1,
    )


def gp_ucb_logits(means, stds, beta, gamma):
    return beta * (means.float() + torch.exp(gamma) * stds.float())


def lookup_table_logits(trial_ids, table):
    return table[trial_ids.long()]
