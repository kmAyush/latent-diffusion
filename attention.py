import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):

    def __init__(self, n_heads:int, d_embed:int, in_proj_bias=True, out_proj_bias=True):
        super()
        # Projecting input to query, key, and value
        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias=in_proj_bias)

        # Projecting concatenated heads back to d_embed dimensions
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x:torch.Tensor, causal_mask = False):
        # X : (batch_size, seq_len, d_embed)

        input_shape = x.shape
        batch_size, seq_length, d_embed = input_shape

        intermin_shape  = (batch_size, seq_length, self.n_heads, self.d_head)

        # q, k, v shape : (batch_size, seq_length, d_embed)
        q, k, v = self.in_proj(x).chunks(3, dim = -1)

        # Moving head dimension to 2nd dimension
        # q, k, v shape after reshape: (batch_size, n_heads, seq_length, d_head)
        q = q.view(intermin_shape).transpose(1, 2)
        k = k.view(intermin_shape).transpose(1, 2)
        v = v.view(intermin_shape).transpose(1, 2)

        # attn_weight shape: (batch_size, n_heads, seq_length, seq_length)
        attn_weight = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(attn_weight, dtype=torch.bool).triu(1)
            attn_weight.masked_fill(mask, -torch.inf)

        attn_weight /= math.sqrt(self.d_head)
        attn_weight = F.softmax(attn_weight, dim = -1)

        # output shape after attention: (batch_size, n_heads, seq_length, d_head)
        output = attn_weight @ v
        # output shape after transpose: (batch_size, seq_length, n_heads, d_head)
        output = output.transpose(1,2)
        # output shape after reshape: (batch_size, seq_length, d_embed)
        output = output.reshape(input_shape)
        output = self.out_proj(output)

        return output


class CrossAttention(nn.Module):

    def __init__(self, n_heads:int, d_embed:int, d_cross:int, in_proj_bias=True, out_proj_bias=True):
        super()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)

        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_heads = d_embed // n_heads


    def forward(self, latent, context):
        # latent  : (batch_size, seq_len_q, dim_q)
        # context : (batch_size, seq_len_kv, dim_kv) * (batch_size, 77, 768)

        input_shape = latent.shape
        batch_size, seq_len, d_embed = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(latent)
        k = self.k_proj(context)
        v = self.v_proj(context)

        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)

        weight = q @ k.transpose(-1,-2)
        weight /= math.sqrt(self.d_head)
        weight = weight @ v

        output = output.transpose(1,2).continuous()
        output = output.view(input_shape)
        output = self.out_proj(output)

        return output

