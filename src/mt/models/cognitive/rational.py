
import torch
import torch.nn as nn
from mt.models.cognitive.preprocessing import preprocess_rational_data





class RationalModel(nn.Module):
    required_columns = ['choice', 'ground_truth']
    def __init__(self,dim:int):
        super().__init__()
        self.rational_params=nn.Parameter(torch.randn(dim,dim))
        self.ignore_index=-100
        
    def preprocess_data(self,train_df,eval_df):
        return preprocess_rational_data(train_df, eval_df, ignore_index=self.ignore_index)

    def forward(self,data):
        action=data['ground_truth'].long()
        return self.rational_params[action]
  
    
