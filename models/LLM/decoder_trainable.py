import torch
import torch.nn as nn
import torch.nn.functional as F

class TrainableDecoder(nn.Module):
  def __init__(self,vocab_size,d_model,d_ff,max_seq_len,num_heads):
    super(TrainableDecoder,self).__init__()

    self.token_embedding=nn.Embedding(vocab_size, d_model)
    self.position_embedding=nn.Embedding(max_seq_len,d_model)
    self.norm1=nn.LayerNorm(d_model)

    self.attn=nn.MultiheadAttention(
      d_model,num_heads,
      batch_first=True,
      dropout=0.1
    )

    self.ffn=nn.Sequential(
        nn.Linear(d_model,d_ff),
        nn.ReLU(),
        nn.Linear(d_ff,d_model)
    )

    self.norm2=nn.LayerNorm(d_model)
    self.output=nn.Linear(d_model,vocab_size)


  def forward(self,x):
    batch_size,seq_len=x.shape
    position_ids=torch.arange(seq_len,device=x.device).unsqueeze(0)
    position_emb=self.position_embedding(position_ids)
    print(position_emb.shape)
    x=self.token_embedding(x)
    x=x+position_emb
    print(f"x shape is: {x.shape}")
    x_norm=self.norm1(x)
    print(f"x shape is: {x.shape}")
    causal_mask = torch.triu(torch.ones(seq_len,seq_len,device=x.device),diagonal=1).bool()
    context,_=self.attn(x_norm,x_norm,x_norm,is_causal=True,attn_mask=causal_mask)
    x=x+context
    x=x+self.ffn(self.norm2(x))
    logits=self.output(x)
    return logits


batch_size = 2
seq_len = 5
vocab_size = 100
d_model = 64
d_ff = 128

model = TrainableDecoder(vocab_size,d_model,d_ff,10,num_heads=8)
print(model)
input = torch.randint(0,vocab_size,(batch_size,seq_len))
output = model(input)
print(output.shape)
