import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()
        # Embedding prompt
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        # Embedding position
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))
    
    def forward(self, tokens):
        x = self.token_embedding(tokens)
        x += self.position_embedding

        # (batch_size, seq_len, dim)
        # (2, 77, 768)
        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        
        self.layernorm_1 = nn.LayerNorm(n_embd)
        # Self attention
        self.attention = SelfAttention(n_head, n_embd)

        # Before feedforward layer
        self.layernorm_2 = nn.LayerNorm(n_embd)

        # Feedforward layer
        # Increase hidden dimension to 4 times the embedding dimension. 
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        # Decrease hidden dimension back to original embedding dimension. 
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        # (Batch_Size, Seq_Len, Dim)
        # (2, 77, 768)
        residue = x
        
        # Apply self-attention layer
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue

        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        # QuickGELU activation function
        x = x * torch.sigmoid(1.702 * x)   
        x = self.linear_2(x)
        x += residue

        # (Batch_Size=2, Seq_Len=77, Dim=768)
        return x

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])
        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # tokens : (Batch_Size=3, Seq_Len=77)
        tokens = tokens.type(torch.long)
        
        # state : (Batch_Size=3, Seq_Len=77, Dim=768)
        state = self.embedding(tokens)

        for layer in self.layers: 
            state = layer(state)

        output = self.layernorm(state)

        # output : (Batch_Size=3, Seq_Len=77, Dim=768)
        return output