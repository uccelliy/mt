"""Lookup-table choice model."""

from __future__ import annotations

import torch
import torch.nn as nn

from mt.models.cognitive.formula_base import FormulaOnlyCognitiveModel


def lookup_table_logits(trial_ids, table):
    return table[trial_ids.long()]


class LookupTableModel(FormulaOnlyCognitiveModel):
    def __init__(self, num_trials: int, num_options: int):
        super().__init__()
        self.logits = nn.Parameter(0.01 * torch.randn(num_trials, num_options))

    def forward(self, data):
        return lookup_table_logits(data["trial"].long(), self.logits)
