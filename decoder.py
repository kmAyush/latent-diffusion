import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention
from vae_attention import VAE_AttentionBlock
from vae_residual import VAE_ResidualBlock

class VAE_Decoder(nn.Sequential):

    def __init__(self):
        super(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            # FROM (batch_size=3, channel=4, height=64, width=64)
            # TO   (batch_size=3, channel=512, height=64, width=64)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            # FROM (batch_size=3, channel=512, height=64, width=64)
            # TO   (batch_size=3, channel=512, height=128, width=128)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # FROM (batch_size=3, channel=512, height=128, width=128)
            # TO   (batch_size=3, channel=512, height=256, width=256)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            # FROM (batch_size=3, channel=256, height=256, width=256)
            # TO   (batch_size=3, channel=256, height=512, width=512)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),
            nn.SiLU(),

            # FROM (batch_size=3, channel=128, height=64, width=64)
            # TO   (batch_size=3, channel=3, height=64, width=64)
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        
        #X : batch_size=3, channel=4, height=64, width=64
        # Remove constant term from x
        x /=0.18215

        for module in self:
            x = module(x)

        #X : batch_size=3, channel=3, height=512, width=512
        return x
