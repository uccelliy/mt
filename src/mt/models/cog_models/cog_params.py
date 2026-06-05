import torch.nn as nn
import torch


class Temperature(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(0.01 * torch.randn([]))

    def forward(self, values):
        """
        Multiplies the input tensor with a parameter.

        Parameter
        ---------
        values : tensor of any shape.

        Returns
        -------
        tensor of any shape
        Scaled tensor.
        """

        return values * self.beta

class Stickiness(nn.Module):
    def __init__(self, num_options):
        super().__init__()
        self.num_options = num_options
        self.beta = nn.Parameter(0.01 * torch.randn([]))

    def forward(self, choices):
        """
        Returns whether a choice has been selected on the previous trial.

        Parameter
        ---------
        choices : tensor of shape (N, T).

        Returns
        -------
        tensor of shape (N, T, self.num_options)
        Tensor filled with ones if choices has been selected on the previous trial.
        """

        num_tasks = choices.shape[0]

        previous_choices_0 = torch.zeros(num_tasks, 1, self.num_options)
        previous_choices_1 = torch.stack([(choices[:, :-1] == a).float() for a in range(self.num_options)], dim=-1)
        previous_choices = torch.cat([previous_choices_0, previous_choices_1], dim=1)

        return previous_choices * self.beta

class InformationBonus(nn.Module):
    def __init__(self, num_options):
        super().__init__()
        self.num_options = num_options
        self.beta = nn.Parameter(0.01 * torch.randn([]))

    def forward(self, choices):
        """
        Returns how often a choice has been chosen up to trial t.

        Parameter
        ---------
        choices : tensor of shape (N, T).

        Returns
        -------
        tensor of shape (N, T, self.num_options)
        Tensor containing how often a choice has been chosen up to trial t.
        """

        num_tasks = choices.shape[0]

        cumsum_choices_0 = torch.zeros(num_tasks, 1, self.num_options)
        cumsum_choices_1 = torch.stack([torch.cumsum((choices[:, :-1] == a).float(), dim=1) for a in range(self.num_options)], dim=-1)
        cumsum_choices = torch.cat([cumsum_choices_0, cumsum_choices_1], dim=1)

        return cumsum_choices * self.beta