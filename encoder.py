import torch
from torch import nn
from torch.nn import functional as F
from vae_attention import VAE_AttentionBlock
from vae_residual import VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        # DOWNSAMPLING

        super(
            # (batch_size=3, channel=128, height=512, width=512)
            nn.Conv2d(3, 128, kernel_size = 3, padding=1),

            VAE_ResidualBlock(128, 128),

            # FROM (batch_size=3, channel=128, height=512, width=512)
            # TO   (batch_size=3, channel=128, height=512, width=512)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # FROM (batch_size=3, channel=128, height=512, width=512)
            # TO   (batch_size=3, channel=256, height=256, width=256)
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),

            # FROM (batch_size=3, channel=256, height=256, width=256)
            # TO   (batch_size=3, channel=256, height=128, width=128)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # FROM (batch_size=3, channel=256, height=128, width=128)
            # TO   (batch_size=3, channel=512, height=128, width=128)
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),

            # FROM (batch_size=3, channel=512, height=128, width=128)
            # TO   (batch_size=3, channel=512, height=64, width=64)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            nn.GroupNorm(32, 512),
            nn.SiLU(),

            # FROM (batch_size=3, channel=512, height=64, width=64)
            # TO   (batch_size=3, channel=8, height=64, width=64)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # linear layer
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x:torch.Tensor, noise:torch.Tensor) -> torch.Tensor:

        for module in self:
            if getattr(module, 'stride', None) == (2,2):
                # (pad_left:0, pad_right:1, pad_top:0, pad_bottom:1) 
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        
        # FROM (batch_size=3, channel=8, height=64, width=64)
        # TO   (batch_size=3, channel=4, height=64, width=64)
        # TO   (mean, log_variance)
        mean, log_variance = torch.chunk(x, 2, dim = 1)

        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        standard_dev = variance.sqrt()

        # Transforming from N(0,1) to N(mean, variance)
        x = mean + standard_dev * noise

        # From paper
        x *= 0.18215

        return x