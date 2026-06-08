"""Odd-one-out model."""

from __future__ import annotations

import torch
import torch.nn as nn

from mt.models.cognitive._base import FormulaOnlyCognitiveModel
from mt.models.cognitive._formulas._choice import odd_one_out_logits


class OddOneOutModel(FormulaOnlyCognitiveModel):
    config_keys = ("num_objects", "embedding_dim")

    def __init__(self, num_objects: int, embedding_dim: int = 16):
        super().__init__()
        self.num_objects = num_objects
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Parameter(0.01 * torch.randn(num_objects, embedding_dim))

    def compute_logits(self, data):
        if "option_embeddings" in data:
            option_embeddings = data["option_embeddings"]
        else:
            option_embeddings = self.embeddings[data["object_ids"].long()]
        return odd_one_out_logits(option_embeddings)
