import torch
import torch.nn as nn


class NoiseCeiling(nn.Module):
    def __init__(self, UID, num_questions=14568, num_options=2, ):
        super().__init__()
        self.param_tensor = nn.Parameter(torch.randn(num_questions, num_options))
        self.UID = UID

    def preprocess_data(self, train_df, eval_df):
        train_data = {}

        mapping_dict = {k: v for k, v in zip(train_df[self.UID], train_df[self.UID].astype('category').cat.codes)}
        print(mapping_dict)

        train_data['choice'] = torch.from_numpy(train_df['choice'].values)
        train_data[self.UID] = torch.from_numpy(train_df[self.UID].map(mapping_dict).values).long()


        eval_data = {}
        eval_data['choice'] = torch.from_numpy(eval_df['choice'].values)
        eval_data[self.UID] = torch.from_numpy(eval_df[self.UID].map(mapping_dict).values).long()

        return train_data, eval_data

    def forward(self, data):
        params = self.param_tensor[data[self.UID]]
        return params
