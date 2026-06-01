import torch
import torch.nn as nn
import numpy as np

from mt.data.data_provider import pd_to_pth
from mt.models.cog_model.cog_params import Temperature


class DualSystemsModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.ignore_index = -100

        self.tau = nn.Parameter(torch.randn([]))
        self.alpha = nn.Parameter(torch.randn([]))
        self.lambd = nn.Parameter(torch.randn([]))
        self.stickiness = nn.Parameter(torch.randn([]))

        self.value_logits = Temperature()

    def preprocess_data(self, train_df, eval_df):

        train_df['choice'] = train_df['choice'].replace(2, -1)
        eval_df['choice'] = eval_df['choice'].replace(2, -1)
        train_df = train_df.replace(-1, np.nan)
        eval_df = eval_df.replace(-1, np.nan)

        for participant in train_df['participant'].unique():
            df_p = train_df[train_df['participant'] == participant]

        train_df_step1 = train_df[train_df['current_state'] == 999]
        train_df_step2 = train_df[train_df['current_state'] != 999]

        for participant in train_df_step1['participant'].unique():
            df_p1 = train_df_step1[train_df_step1['participant'] == participant]
            df_p2 = train_df_step2[train_df_step2['participant'] == participant]

        eval_df_step1 = eval_df[eval_df['current_state'] == 999]
        eval_df_step2 = eval_df[eval_df['current_state'] != 999]

        train_data1 = pd_to_pth(train_df_step1, ['current_state', 'reward', 'choice'], keys=['participant', 'trial'])
        train_data2 = pd_to_pth(train_df_step2, ['current_state', 'reward', 'choice'], keys=['participant', 'trial'])

        eval_data1 = pd_to_pth(eval_df_step1, ['current_state', 'reward', 'choice'], keys=['participant', 'trial'])
        eval_data2 = pd_to_pth(eval_df_step2, ['current_state', 'reward', 'choice'], keys=['participant', 'trial'])

        train_data = {}
        eval_data = {}
        for key in train_data1.keys():
            train_data[key] = torch.stack([train_data1[key], train_data2[key]], dim=-1)
            eval_data[key] = torch.stack([eval_data1[key], eval_data2[key]], dim=-1)

        train_data['choice'] = torch.nan_to_num(train_data['choice'], nan=self.ignore_index).long()
        eval_data['choice'] = torch.nan_to_num(eval_data['choice'], nan=self.ignore_index).long()

        return train_data, eval_data

    def forward(self, data):
        logits = self.forward_two_step(data)
        return logits

    def forward_two_step(self, data):
        tau = torch.sigmoid(self.tau)
        alpha = torch.sigmoid(self.alpha)
        lambd = torch.sigmoid(self.lambd)
        stickiness = torch.tanh(self.stickiness)

        action_1 = data['choice'][:, :, 0].long()
        action_2 = data['choice'][:, :, 1].long()
        state = data['current_state'][:, :, 1].long()
        reward = data['reward'][:, :, 1]

        transition_matrix = torch.Tensor([[0.7, 0.3], [0.3, 0.7]])
        n_participants = data['choice'].shape[0]
        n_trials = data['choice'].shape[1]
        action_repeat = torch.zeros(2)

        logits = torch.zeros(n_participants, n_trials, 2, 2)
        for par in range(n_participants):
            q_mf = torch.zeros(3, 2)  # initialise model free values
            for trial in range(n_trials):

                max_q, _ = torch.max(q_mf[1:], dim=1)
                q_mb = transition_matrix @ max_q
                q_net = tau * q_mb.clone() + (1 - tau) * q_mf[0].clone()

                if (self.ignore_index == action_1[par, trial].item()): #, action_2[par, trial].item()
                    continue

                logits[par, trial, 0] = q_net + action_repeat * stickiness
                logits[par, trial, 1] = q_mf[state[par, trial] + 1]

                if not (self.ignore_index == action_2[par, trial].item()):
                    delta_1 = q_mf[state[par, trial] + 1, action_2[par, trial]] - q_mf[0, action_1[par, trial]]
                    q_mf[0, action_1[par, trial]] += alpha * delta_1
                    delta_2 = reward[par, trial] - q_mf[state[par, trial] + 1, action_2[par, trial]]
                    q_mf[state[par, trial] + 1, action_2[par, trial]] += alpha * delta_2
                    q_mf[0, action_1[par, trial]] += lambd * alpha * delta_2

                action_repeat = torch.zeros(2)
                action_repeat[action_1[par, trial]] = 1

        return self.value_logits(logits)
