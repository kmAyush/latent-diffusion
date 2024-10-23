import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    # Attention block allows the model to focus on important region of the image.
    
    def __init__(self, channels:int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        residue = x

        x = self.groupnorm(x)
        batch_size, features, height, width = x.shape
    
        x = x.view((batch_size, features, height*width))

        # FROM (batch_size, channel, height * width)
        # TO   (batch_size, height * width, channel)
        x = x.transpose(-1, -2)

        x = self.attention(x)
        x = x.transpose(-1, -2)
        x = x.view((batch_size, features, height, width))

        x += residue
        return x