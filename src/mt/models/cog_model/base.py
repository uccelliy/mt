import torch.nn as nn
import torch

class base(nn.Module):
    def __init__(self):
        super().__init__()


    def preprocess_data(self,train_df,eval_df):
        pass