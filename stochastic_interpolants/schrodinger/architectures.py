import math
import numpy as np
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
        
    def forward(self, x, t):
        if isinstance(t, int) or len(t.shape) < 2:
            t = torch.ones_like(x[..., 0].unsqueeze(-1) * t)
        x = torch.cat([x, t], dim=-1)
        return self.model(x)