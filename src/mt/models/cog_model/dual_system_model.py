import torch
import torch.nn as nn

from mt.models.cog_model.preprocessing import preprocess_dual_system_data
from mt.models.cog_model.cog_params import Temperature


class DualSystemsModel(nn.Module):
    required_columns = ['current_state', 'reward', 'choice']
    def __init__(self):
        super().__init__()

        self.ignore_index = -100

        self.tau = nn.Parameter(torch.randn([]))
        self.alpha = nn.Parameter(torch.randn([]))
        self.lambd = nn.Parameter(torch.randn([]))
        self.stickiness = nn.Parameter(torch.randn([]))

        self.value_logits = Temperature()

    def preprocess_data(self, train_df, eval_df):
        return preprocess_dual_system_data(train_df, eval_df, ignore_index=self.ignore_index)

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
