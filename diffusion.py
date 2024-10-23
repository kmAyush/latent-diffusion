import torch
from torch import nn
from torch.nn import functional as F
from unet import UNET


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        # x: (1, 320)

        # x: (1, 1280)
        x = self.linear_1(x)
        x = F.silu(x) 
        x = self.linear_2(x)

        return x

class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        return x
    

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
    
    def forward(self, latent, context, time):
        # latent: (Batch_Size=2, 4, Height / 8 = 64, Width / 8 = 64)
        # context: (Batch_Size=2, Seq_Len=77, Dim=768)
        # time: (1, 320)

        # time : (1, 1280)
        time = self.time_embedding(time)

        # FROM : (batch=2, channel=4,   Height/8 = 64, Width/8 = 64) 
        # To   : (batch=2, channel=320, Height/8 = 64, Width/8 = 64)
        output = self.unet(latent, context, time)

        # FROM : (Batch=2, channel=320, Height/8 = 64, Width/8 = 64) 
        # TO   : (Batch=2, channel=4,   Height/8 = 64, Width/8 = 64) 
        output = self.final(output)
        
        return output