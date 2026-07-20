"""Legacy metric callbacks used only by the Centaur reference experiments."""

from __future__ import annotations

import torch


def full_log_likelihoods(logits, labels):
    with torch.no_grad():
        logits = logits.float().cpu()
        labels = labels.cpu()
        labels = torch.cat((labels[0, 1:], -100 * torch.ones(1).long()), 0)
        logits = logits[0]
        cross_entropy = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        total_loss = []
        item_loss = 0
        item_counter = 0
        for index in range(cross_entropy.shape[0]):
            if labels[index] != -100:
                item_loss += cross_entropy[index]
                item_counter += 1
            elif item_counter != 0:
                total_loss.append(item_loss)
                item_loss = 0
                item_counter = 0
        return torch.tensor(total_loss)


def compute_metrics_mean(pred):
    print(pred.predictions.shape, flush=True)
    return {"custom_loss": pred.predictions.mean()}


def compute_metrics(pred):
    return {"custom_loss": pred.predictions}
