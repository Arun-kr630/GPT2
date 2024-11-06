import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim:int=4098
    n_layer:int=32
    n_heads:int=32
    n_kv_heads:Optional[int]=None
    vocab_size:int=50304
    norm_eps:float=1e-5

    max_batch_size:int=32
    max_seq_len:int=2048
    device:str='cpu'


def precompute_theta_pos_frequencies(head_dim,seq_len,device,theta:float=10000.0):
    assert head_dim%2==0, "Head_dim must be divisible by 2"

    theta_numerator=torch.arange(0,head_dim,2).float()
    theta=1.0/(theta ** (theta_numerator/head_dim)).to(device)

    m=torch.arange(seq_len,device=device)
    freqs=torch.outer(m,theta).float()
    freqs_complex=torch.polar(torch.ones_like(freqs),freqs)

    return freqs_complex

def apply_rotary_embeddings(x, freqs_complex, device: str):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    freqs_complex=freqs_complex.to(device)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


class FeedForward(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
        hidden_dim=4 * args.dim
        self.w1=nn.Linear(args.dim,hidden_dim,bias=False)
        self.w2=nn.Linear(args.dim,hidden_dim,bias=False)
        self.w3=nn.Linear(hidden_dim,args.dim,bias=False)
    def forward(self,x):
        x1=self.w1(x)
        x2=self.w2(x)
        hidden=F.silu(x1) * x2
        return self.w3(hidden)

class SelfAttention(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
        self.n_kv_heads=args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_head_q=args.n_heads
        self.n_rep=self.n_head_q // self.n_kv_heads
        self.head_dim=args.dim//args.n_heads

        self.wq=nn.Linear(args.dim,args.dim)
        self.wk=nn.Linear(args.dim,args.dim)
        self.wv=nn.Linear(args.dim,args.dim)
        self.wo=nn.Linear(args.dim,args.dim)

    def forward(self,x,freqs_complex):

        batch_size,seq_len,_=x.shape
        xq=self.wq(x)
        xk=self.wk(x)
        xv=self.wv(x)

        xq=xq.view(batch_size,seq_len,self.n_head_q,self.head_dim)
        xk=xk.view(batch_size,seq_len,self.n_head_q,self.head_dim)
        xv=xv.view(batch_size,seq_len,self.n_head_q,self.head_dim)
        
        xq=apply_rotary_embeddings(xq,freqs_complex,device=x.device)
        xk=apply_rotary_embeddings(xk,freqs_complex,device=x.device)
        xq=xq.transpose(1, 2)
        xk=xk.transpose(1, 2)
        xv=xv.transpose(1,2)
        y=F.scaled_dot_product_attention(xq,xk,xv,is_causal=True)

        y=y.transpose(-1,-2).contiguous().view(batch_size,seq_len,-1)
        y=self.wo(y)
        return y
        



class EncoderBlock(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
        self.args=args
        self.head_dim=args.dim//args.n_heads

        self.attention=SelfAttention(args)
        self.feed_forward=FeedForward(args)
        self.attention_norm=nn.RMSNorm(args.dim)
        self.ffn_norm=nn.RMSNorm(args.dim)

    def forward(self,x,freqs_complex):
        h=x+self.attention(self.attention_norm(x),freqs_complex)
        out=h+self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
        self.args=args
        self.tok_embeddings=nn.Embedding(args.vocab_size,args.dim)
        self.layers=nn.ModuleList([EncoderBlock(args) for _ in range(args.n_layer)])
        self.norm=nn.RMSNorm(args.dim)
        self.output=nn.Linear(args.dim,args.vocab_size,bias=False)

        self.freqs_complex=precompute_theta_pos_frequencies(args.dim//args.n_heads,args.max_seq_len * 2,device=args.device)

    def forward(self,tokens):
        batch_size,seq_len=tokens.shape
        h=self.tok_embeddings(tokens)
        freqs_complex=self.freqs_complex[:seq_len]

        for layer in self.layers:
            h=layer(h,freqs_complex)
        h=self.norm(h)
        output=self.output(h)
        return output
    

device='cpu'
if torch.cuda.is_available():
    device='cuda:2'
print(f"using device {device}")


import numpy as np
import os
import torch
def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) 
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B1"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
 
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T 

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T 
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T
        return x,y
B = 26 
T = 1024 
from tqdm import tqdm
train_loader = DataLoaderLite(B=B, T=T, split="train")



    
model=Transformer(ModelArgs())
model.to(device)
criterian=nn.CrossEntropyLoss()
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(0.9,0.95))
for epoch in range(10):
    for i in tqdm(range(6414)):
        x,y = train_loader.next_batch()
        x,y=x.to(device),y.to(device)
        optimizer.zero_grad()    
        logits=model(x)
        loss=criterian(logits.view(B*T,-1),y.view(-1))
        loss.backward()
        optimizer.step()
        if (i%100) == 0:
            print(f"step {i}, loss {loss.item()}")
    checkpoint={"model_state":model.state_dict()}
    torch.save(checkpoint,f"llama_7b_checks_{epoch}.pth")

