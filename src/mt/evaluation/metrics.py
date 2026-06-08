"""Evaluation metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def choice_nll(logits, choices):
    """Compute mean negative log-likelihood for choice logits."""

    return F.cross_entropy(logits.flatten(0, -2), choices.flatten().long())


def full_log_likelihoods(logits, labels):
    with torch.no_grad():
        logits = logits.float().cpu()
        labels = labels.cpu()
        labels = torch.cat((labels[0, 1:], -100 * torch.ones(1).long()), 0)
        logits = logits[0]
        ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        total_loss = []
        item_loss = 0
        item_counter = 0
        for i in range(ce.shape[0]):
            if labels[i] != -100:
                item_loss += ce[i]
                item_counter += 1
            else:
                if item_counter != 0:
                    total_loss.append(item_loss)
                    item_loss = 0
                    item_counter = 0
        return torch.Tensor(total_loss)


def compute_metrics_mean(pred):
    print(pred.predictions.shape, flush=True)
    return {"custom_loss": pred.predictions.mean()}


def compute_metrics(pred):
    return {"custom_loss": pred.predictions}
