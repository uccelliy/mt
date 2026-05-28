import math
from os import error

import pandas as pd
import numpy as np
import torch 
from torch import nn

class DataGenerator:
    def __init__(self):
        pass
    
    def get_data(self):
        pass
    
    def load_data(self,file_name):
        raw_data=pd.read_csv(file_name,header=None)

        
    
    def generate_data(self,task_name):
        if task_name == "GO":
            T_mod = torch.randint(1,2,(1,0))
            T_stim1 = torch.randint(500, 1500,(1,0))
            T_stim2 = 0
            T_delay = 0
        elif task_name == "RTGO":
            T_mod = torch.randint(1,2,(1,0))
            T_stim1 = torch.randint(500, 2500,(1,0))
            T_stim2 = 0
            T_delay = 0
        elif task_name == "DlyGO":
            T_mod = torch.randint(1,2,(1,0))
            T_stim1 = torch.randint(500, 1500,(1,0))
            T_stim2 = 0
            T_delay = torch.tensor([200,400,800,1600])[torch.randint(0,4,(1,0))]
        else :
            raise error

    def process_data(self,raw_data):
        pass


def compute_activation(activation_func_name, activation_input):
    if activation_func_name == "Tanh":
        return torch.tanh(activation_input)
    elif activation_func_name == "ReLU":
        return torch.relu(activation_input)
    elif activation_func_name == "ReTanh":
        return torch.relu(torch.tanh(activation_input))
    else :
        return torch.nn.functional.softplus(activation_input)


class NeuralRNN(nn.Module):
    def __init__(self,x,y,alpha,alpha_in,q_rec,alpha_rac,activation_func_name):
        super().__init__()
        self.activation_func_name = activation_func_name
        self.x = x + torch.sqrt(2/alpha)*alpha_in*torch.randn(len(x))
        self.y = y
        self.alpha = alpha
        self.h_t = torch.zeros(256,1)
        self.h_t_1 = self.h_t
        self.W_rec = q_rec * torch.eye(256)
        self.W_in = torch.normal(0,1/math.sqrt(len(x)),size=(256,len(x)))
        self.W_out = torch.normal(0,1/math.sqrt(256),size=(len(y),256))
        self.bias = torch.sqrt(2*torch.inverse(alpha)*alpha_rac**2)*torch.randn(256,1)
        
    def stimuli_output(self):
        output = 1/(1+torch.exp(-(self.h * self.W_out)))

    def forward(self):
        activation_input = self.W_rec*self.h_t_1+self.W_in*self.x+self.bias
        activation = compute_activation(self.activation_func_name,activation_input)
        self.h_t = (1-self.alpha) * self.h_t_1 + self.alpha * activation
    
 