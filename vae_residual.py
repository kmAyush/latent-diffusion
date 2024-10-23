import torch
from torch import nn
from torch.nn import functional as F

class VAE_ResidualBlock(nn.Module):
    # Residual block enhances deep network learning while preserving information.
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # If the number of input channels is equal to the output channels, 
        # the residual connection is an identity mapping (no change in dimensions).
        # Otherwise, a 1x1 convolution is applied to match the dimensions.
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        residue = x
        
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return (x + self.residual_layer(residue))