"""Lookup-table choice model."""

from __future__ import annotations

import torch
import torch.nn as nn

from mt.models.cognitive._base import FormulaOnlyCognitiveModel


def lookup_table_logits(trial_ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    return table[trial_ids.long()]


class LookupTableModel(FormulaOnlyCognitiveModel):
    config_keys = ("num_trials", "num_options")

    def __init__(self, num_trials: int, num_options: int):
        super().__init__()
        self.num_trials = num_trials
        self.num_options = num_options
        self.logits = nn.Parameter(0.01 * torch.randn(num_trials, num_options))

    def compute_logits(self, data):
        return lookup_table_logits(data["trial"].long(), self.logits)
