import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer

class DecoderForUnderstanding(nn.Module):
  def __init__(self,vocab_size,d_model,d_ff,head_num):
    super(DecoderForUnderstanding, self).__init__()
    self.embedding_weight=nn.Parameter(torch.randn(vocab_size,d_model))
    multi_head_att_dim=d_model//head_num
    self.Query_weight=nn.ParameterList([
        nn.Parameter(torch.randn(d_model,multi_head_att_dim))
        for _ in range(head_num)
        ])
    self.Key_weight=nn.ParameterList([
        nn.Parameter(torch.randn(d_model,multi_head_att_dim))
        for _ in range(head_num)
        ])
    self.Value_weight=nn.ParameterList([
        nn.Parameter(torch.randn(d_model,multi_head_att_dim))
        for _ in range(head_num)
        ])
    self.ml_att_weight=nn.Parameter(torch.randn(d_model,d_model))
    self.FFN1_weight=nn.Parameter(torch.randn(d_model,d_ff))
    self.FFN1_bias=nn.Parameter(torch.randn(d_ff))
    self.FFN2_weight=nn.Parameter(torch.randn(d_ff,d_model))
    self.FFN2_bias=nn.Parameter(torch.randn(d_model))
    self.output_weight=nn.Parameter(torch.randn(d_model,vocab_size))
    self.output_bias=nn.Parameter(torch.randn(vocab_size))

  def attention(self,query,key,value):
    d_scaler=(query.size(-1))**0.5
    attention_score=torch.matmul(query,key.transpose(1,2))/d_scaler
    print(f"attention_score shape is: {attention_score.shape}")
    casual_mask=torch.triu(torch.ones(attention_score.shape[1],attention_score.shape[2]),diagonal=1).bool()
    print(f"casual is: {casual_mask}")
    attention_score=attention_score.masked_fill(casual_mask,float("-inf"))
    print(f"attention_score is: {attention_score}")
    attention_weight=torch.softmax(attention_score, dim=-1)
    print(f"attention_weight shape: {attention_weight}")
    context=torch.matmul(attention_weight,value)
    #print(f"context is: {context}")
    return context

  def multihead_attention(self,x,query_list,key_list,value_list):
    context_list=[]
    for query_w,key_w,value_w in zip(query_list,key_list,value_list):
      query=torch.matmul(x,query_w)
      print(f"query shape is: {query.shape}")
      key=torch.matmul(x,key_w)
      print(f"key shape is: {key.shape}")
      value=torch.matmul(x,value_w)
      print(f"value shape is: {value.shape}")
      context=self.attention(query,key,value)
      context_list.append(context)

    ml_context=torch.matmul(torch.cat(context_list,dim=-1),self.ml_att_weight)
    return ml_context


  def position_embedding(self,x):
    pass

  def embedding(self,x):
    return self.embedding_weight[x]

  def ffn(self,x):
    x=torch.matmul(x,self.FFN1_weight)+self.FFN1_bias

    x=F.relu(x)
    print(f"relu x shape is: {x.shape}")
    x=torch.matmul(x,self.FFN2_weight)+self.FFN2_bias

    return x


  def Norm(self,x):
    return (x-torch.mean(x,dim=-1,keepdim=True))/torch.std(x,dim=-1,keepdim=True)

  def forward(self,x):
    x=self.embedding(x)
    print(f"x shape is: {x.shape}")
    x=self.Norm(x)
    print(f"x shape is: {x.shape}")
    context=self.multihead_attention(x,self.Query_weight,self.Key_weight,self.Value_weight)
    print(f"context shape is: {context.shape}")
    x=x+context
    x= self.ffn(x)
    x=self.Norm(x)+x
    output=torch.matmul(x,self.output_weight)+self.output_bias
    print(f"output shape is: {output.shape}")
    return output


batch_size = 2
seq_len = 5
vocab_size = 100
d_model = 64
d_ff = 128

model = DecoderForUnderstanding(vocab_size, d_model, d_ff)
input = torch.randint(0,vocab_size,(batch_size,seq_len))
output = model(input)
print(output.shape)




