import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention
from vae_attention import VAE_AttentionBlock
from vae_residual import VAE_ResidualBlock


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            # Increase channel size
            # FROM (batch_size=3, channel=4, height=64, width=64)
            # TO   (batch_size=3, channel=512, height=64, width=64)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            
        
            VAE_ResidualBlock(512, 512), 
            VAE_AttentionBlock(512), 
            VAE_ResidualBlock(512, 512), 
            
            
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            
            # Resizing image height and weight to double its size, ie, 64 -> 128
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            
            # Resizing image height and weight to double its size, ie, 128 -> 256
            nn.Upsample(scale_factor=2), 
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            
            VAE_ResidualBlock(512, 256), 
            VAE_ResidualBlock(256, 256), 
            VAE_ResidualBlock(256, 256), 
            
            # Resizing image height and weight to original size, ie, 256 -> 512
            nn.Upsample(scale_factor=2), 
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            
            # FROM (batch_size=3, channel=256, height=512, width=512)
            # TO   (batch_size=3, channel=128, height=512, width=512)
            VAE_ResidualBlock(256, 128), 
            VAE_ResidualBlock(128, 128), 
            VAE_ResidualBlock(128, 128), 
            
            
            nn.GroupNorm(32, 128), 
            nn.SiLU(), 
            
            # FROM (batch_size=3, channel=128, height=512, width=512)
            # TO   (batch_size=3, channel=3, height=64, width=64)
            nn.Conv2d(128, 3, kernel_size=3, padding=1), 
        )

    def forward(self, x):
        # x : batch_size=3, channel=4, height=64, width=64
        
        # Remove the scaling factor
        x /= 0.18215

        for module in self:
            x = module(x)

        # x : batch_size=3, channel=3, height=512, width=512
        return x