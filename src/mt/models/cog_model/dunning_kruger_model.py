import torch
import torch.nn as nn



class DunningKruger(nn.Module):
    def __init__(self):
        super().__init__()
        self.param_tensor = nn.Parameter(torch.randn(28, 11))

    def preprocess_data(self, train_df, eval_df):
        for i in range(4, 24):
            train_df.loc[train_df['trial'] == i, 'choice'] = train_df[train_df['trial'] == i]['choice'].astype('category').cat.codes
            eval_df.loc[eval_df['trial'] == i, 'choice'] = eval_df[eval_df['trial'] == i]['choice'].astype('category').cat.codes

        normalizer = torch.Tensor([2, 10, 1, 1,
                                   1, 1, 1, 1, 1,
                                   1, 1, 1, 1, 1,
                                   1, 1, 1, 1, 1,
                                   1, 1, 1, 1, 1,
                                   2, 10, 1, 1])

        train_data = {}
        num_train_participants = len(train_df.participant.unique())
        train_data['choice'] = torch.from_numpy(train_df[(train_df['trial'] != 24)]['choice'].values.astype('float'))
        train_data['choice'] = (train_data['choice'] // normalizer.repeat(num_train_participants)).long()

        eval_data = {}
        num_eval_participants = len(eval_df.participant.unique())
        eval_data['choice'] = torch.from_numpy(eval_df[(eval_df['trial'] != 24)]['choice'].values.astype('float'))
        eval_data['choice'] = (eval_data['choice'] // normalizer.repeat(num_eval_participants)).long()

        return train_data, eval_data

    def forward(self, data):
        num_participants = int(data['choice'].shape[0] / 28)
        params = self.param_tensor.repeat(num_participants, 1)

        return params