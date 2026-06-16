import torch
import torch.nn as nn
from mt.models.common._base import BaseCognitiveModel
from mt.models.common._preprocessing import preprocess_rational_data


class RationalModel(BaseCognitiveModel):
    required_columns = ["choice", "ground_truth"]
    config_keys = ("dim",)

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.rational_params = nn.Parameter(torch.randn(dim, dim))
        self.ignore_index = -100

    def preprocess_data(self, train_df, eval_df):
        return preprocess_rational_data(train_df, eval_df, ignore_index=self.ignore_index)

    def compute_logits(self, data):
        action = data["ground_truth"].long()
        return self.rational_params[action]
