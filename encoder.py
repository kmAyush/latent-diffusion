import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        
        super().__init__(
            # Initial Image size: (batch_size=3, channel=3, height=512, width=512)
            # TO (batch_size=3, channel=128, height=512, width=512)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            # Residual block enhances deep network learning while preserving information.
            VAE_ResidualBlock(128, 128),
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
            
            # Attention block allows the model to focus on important region of the image.
            VAE_AttentionBlock(512), 

            VAE_ResidualBlock(512, 512), 
            
            nn.GroupNorm(32, 512), 
            nn.SiLU(), 

            # FROM (batch_size=3, channel=512, height=128, width=128)
            # TO   (batch_size=3, channel=8, height=64, width=64)
            nn.Conv2d(512, 8, kernel_size=3, padding=1), 

            nn.Conv2d(8, 8, kernel_size=1, padding=0), 
        )

    def forward(self, x, noise):
        # x: (Batch_Size, Channel, Height, Width)
        # noise: (Batch_Size, 4, Height / 8, Width / 8)

        for module in self:

            if getattr(module, 'stride', None) == (2, 2):
                # Pad: (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom).
                # asymmetric padding at downsampling
                # x :(Batch_Size, Channel, Height + 1, Width + 1)
                x = F.pad(x, (0, 1, 0, 1))
            
            x = module(x)
        
        # FROM (batch_size=3, channel=8, height=64, width=64)
        # TO   (batch_size=3, channel=4, height=64, width=64)
        # TO   (mean, log_variance)    
        mean, log_variance = torch.chunk(x, 2, dim=1)
        
        # Clamp log variance between -30 and 20, so that the variance is between (circa) 1e-14 and 1e8. 
        # too small, it would imply high uncertainty or noise, make the model ignore meaningful features in the data
        log_variance = torch.clamp(log_variance, -30, 20)

        variance = log_variance.exp()
        stdev = variance.sqrt()
        
        # Transforming from N(0,1) to N(mean, variance)
        x = mean + stdev * noise
        
        # Scale factor value taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        x *= 0.18215
        
        return x
