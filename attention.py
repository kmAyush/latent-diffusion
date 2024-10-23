import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # in_proj represents Wq, Wk and Wv matrices
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        
        # out_proj represents the Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        # x: (Batch_Size, Seq_Len, Dim)

        # (Batch_Size, Seq_Len, Dim)
        input_shape = x.shape 
        batch_size, sequence_length, d_embed = input_shape 

        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head) 

        # q, k, v shape : (batch_size, seq_length, d_embed)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # Moving head dimension to 2nd dimension
        # q, k, v shape after reshape: (batch_size, n_heads, seq_length, d_head)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)


        # SCALED DOT PRODUCT ATTENTION
        # attn_weight shape: (batch_size, n_heads, seq_length, seq_length) 
        # attn_weight = (Batch_Size, H, Seq_Len, Dim / H) @ (Batch_Size, H, Dim / H, Seq_Len)
        weight = q @ k.transpose(-1, -2)
        # Mask future token
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1) 
            weight.masked_fill_(mask, -torch.inf) 
        weight /= math.sqrt(self.d_head) 
        weight = F.softmax(weight, dim=-1) 
        output = weight @ v

        # Adjusting dimension
        output = output.transpose(1, 2) 
        output = output.reshape(input_shape) 
        output = self.out_proj(output) 
        
        # (Batch_Size, Seq_Len, Dim)
        return output

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x, y):
        # x (latent): (batch_Size, seq_len_Q, d_embed) = (3, 77, 768)
        # y (context): (batch_Size, seq_len_KV, d_cross) = (3, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        # q : (Batch_Size, Seq_Len_Q, d_embed)
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        # q: (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        q = q.view(interim_shape).transpose(1, 2) 
        k = k.view(interim_shape).transpose(1, 2) 
        v = v.view(interim_shape).transpose(1, 2) 
        
        # SCALED DOT PRODUCT ATTENTION
        # weight : (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        # output : (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
        output = weight @ v
        
        # .view(), require contiguous tensors (i.e., rearrange the memory to make it contiguous)
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        
        output = self.out_proj(output)

        # (Batch_Size=2, Seq_Len_Q=77, Dim_Q=768)
        return output