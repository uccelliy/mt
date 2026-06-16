import torch
import torch.nn as nn

from mt.models.cognitive._base import BaseCognitiveModel
from mt.models.cognitive._preprocessing import preprocess_dunning_kruger_data


class DunningKruger(BaseCognitiveModel):
    required_columns = ["choice"]

    def __init__(self):
        super().__init__()
        self.param_tensor = nn.Parameter(torch.randn(28, 11))

    def preprocess_data(self, train_df, eval_df):
        return preprocess_dunning_kruger_data(train_df, eval_df)

    def compute_logits(self, data):
        num_participants = int(data["choice"].shape[0] / 28)
        params = self.param_tensor.repeat(num_participants, 1)

        return params
