
import torch
import torch.nn as nn
from mt.data.data_provider import preprocess_rational_data





class RationalModel(nn.Module):
    required_columns = ['choice', 'reward']
    def __init__(self):
        super().__init__()
        self.rational_params=nn.Parameter(torch.randn([]))
        self.ignore_index=-100
        
    def preprocess_data(self,train_df,eval_df):
        return preprocess_rational_data(train_df, eval_df, ignore_index=self.ignore_index)

    def forward(self,data):
        action=data['ground_truth'].long()
        return self.rational_params[action]
  
    
