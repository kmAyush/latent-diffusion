import torch


def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # Time embedding similar as positional encoding in transformers
    # Inspired by sinusoidal positional encoding used in transformers for sequence data.

    # 10000 ^ (-i/160) for i in range(160)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 

    # Convert timestep into a tensor and expand its shape to match the shape of the frequencies.
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    
    # Concatenates both cosine and sine components along last dimension.
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
