import math

import torch
import torch.nn.functional as F

class baseModel(torch.nn.Module):
    def __init__(self, params):
        super(baseModel, self).__init__()
        for k, v in params.items():
            setattr(self, k, v)
        self.freqs = 2. ** torch.linspace(0., self.max_freq, steps=self.max_freq + 1)
        self.in_ch = self.in_ch * (len(self.p_fns) * (self.max_freq + 1) + 1)

    def embed(self, coords):
        return torch.cat([coords, *[getattr(torch, p_fn)(coords * freq) for freq in self.freqs for p_fn in self.p_fns]], -1)
    
