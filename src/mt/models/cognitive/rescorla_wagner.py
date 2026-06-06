import torch
import torch.nn as nn
import torch.nn.functional as F
from mt.models.cognitive.base import BaseCognitiveModel
from mt.models.cognitive.preprocessing import preprocess_rescorla_wagner_data
from mt.models.cognitive.parameters import InformationBonus, Stickiness, Temperature


class RescorlaWagnerModel(BaseCognitiveModel):
    
    required_columns = ['reward', 'choice']
    config_keys = ("num_options",)
    
    def __init__(self, num_options=3):
        super().__init__()

        self.num_options = num_options
        self.ignore_index = -100

        self.value_updating = TabularRescorlaWagnerPlusMinusValueUpdating(num_options)

        self.information_logits = InformationBonus(num_options)
        self.stickiness_logits = Stickiness(num_options)
        self.value_logits = Temperature()

    def preprocess_data(self, train_df, eval_df):
        """
        Preprocesses data into pytorch format.

        Parameter
        ---------
        train_df : pandas dataframe.
        eval_df : pandas dataframe.

        Returns
        -------
        dict, dict
        Dictionaries with tensors named 'choice' and 'reward' of shape (N, T).
        """

        return preprocess_rescorla_wagner_data(
            train_df,
            eval_df,
            ignore_index=self.ignore_index,
        )

    def compute_logits(self, data):
        """
        Model with Rescorla-Wagner-based learning, stickiness and information bonus.

        Parameter
        ---------
        data : dictionary with tensors named 'choice' and 'reward' of shape (N, T).

        Returns
        -------
        tensor of shape (N, T, self.num_options)
        Tensor filled with logits for all options.
        """

        values = self.value_updating(data['choice_for_updating'].long(), data['reward'].float())

        information_logits = self.information_logits(data['choice'].long())
        stickiness_logits = self.stickiness_logits(data['choice'].long())
        value_logits = self.value_logits(values)
        return value_logits + stickiness_logits + information_logits



class TabularRescorlaWagnerPlusMinusValueUpdating(nn.Module):
    def __init__(self, num_options, max_initial_values=100, ignore_index=-100):
        super().__init__()

        self.num_options = num_options
        self.max_initial_values = max_initial_values

        self.alpha_plus = nn.Parameter(0.01 * torch.randn([]))
        self.alpha_minus = nn.Parameter(0.01 * torch.randn([]))
        self.initial_values = nn.Parameter(0.01 * torch.randn([]))

        self.ignore_index = ignore_index

    def forward(self, choices, rewards):
        """
        Performs Rescorla-Wagner updating with separate learning rates for positive and negative prediction errors for the given choices and rewards.

        Parameter
        ---------
        choices : tensor of shape (N, T).
        rewards : tensor of shape (N, T).

        Returns
        -------
        tensor of shape (N, T, self.num_options)
        Tensor filled with estimated values for all options.
        """

        num_tasks = choices.shape[0]
        num_trials = choices.shape[1]

        initial_values = self.max_initial_values * F.tanh(self.initial_values)
        alpha_plus = F.sigmoid(self.alpha_plus)
        alpha_minus = F.sigmoid(self.alpha_minus)

        values = torch.ones(list(choices.shape) + [self.num_options]) * initial_values

        for t in range(num_trials-1):
            # copy over everything
            values[:, t+1, :] = values[:, t, :]
            # compute prediction errors
            prediction_error = rewards[:, t] - values[torch.arange(num_tasks), t, choices[:, t]]
            # zero-out prediction errors for missing trial
            prediction_error[torch.isnan(rewards[:, t])] = 0
            # update values for selected actions
            values[torch.arange(num_tasks), t+1, choices[:, t]] = values[torch.arange(num_tasks), t, choices[:, t]] + (alpha_plus * prediction_error * (prediction_error >= 0).float()) + (alpha_minus * prediction_error * (prediction_error < 0).float())

        return values
