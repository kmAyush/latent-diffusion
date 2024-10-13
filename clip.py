import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class ClipEmbedding(nn.Module):
    def __init__(self, n_vocab:int, n_embed:int, n_tokens:int):
        super()

        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))

    def forward(self, tokens):
        x = self.token_embedding(tokens)
        x += self.position_embedding
        return x

class ClipLayer(nn.Module):
    def __init__(self, n_head:int, n_embed:int):
        super()
        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)
        self.linear_1 = nn.Linear(n_embed, 4*n_embed)
        self.linear_2 = nn.Linear(4*n_embed, n_embed)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x : (batch_size, seq_length, d_embed)

        residue = x
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask = True)
        x += residue

        # FeedForward network
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)

        # apply quickGeLU activation function
        x = x * torch.sigmoid(1.702 * x) 

        x = self.linear(x)
        x += residue
        return x

class Clip(nn.Module):

    def __init__(self):
        self.embedding = ClipEmbedding(49408, 768, 77)

        self.layers = nn.Module([
            ClipLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        output = self.layernorm(state)

        return output
    